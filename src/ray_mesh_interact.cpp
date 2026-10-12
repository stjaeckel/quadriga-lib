// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

#include <complex>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cstdint>

// qd::bits<T> wraps a value of type T (sizeof 1, 2, 4, or 8) and lets you read and write its individual bits
#include "bits.hpp"

// Co-location distance. If 2 faces are closer than that, they are treated as a M2M transition
#define colocation_dist 0.001

// Materials
namespace
{
    // The argument of a refraction root that is real and negative up to rounding, where the root
    // has to be the decaying one.
    //
    // An argument 1 - (e1 / e2) sin^2 is exactly real for two lossless media at a real angle, and
    // the principal root of a negative real with a +0.0 imaginary part is the growing wave, so the
    // sign of the zero is forced. Formed from a recovered complex cosine the same argument is real
    // only up to rounding - the imaginary parts of e1 (1 - c^2) cancel to a few ulp, of either
    // sign - and a test for an exact zero then leaves the branch to that sign: the phase of a
    // total reflection flipped by about 180 deg between neighbouring angles. So the test is
    // relative. A far medium with real loss has a negative imaginary part many orders above the
    // bound and takes the principal root, which is the decaying one, as it always did.
    inline bool qd_evanescent(const std::complex<double> &arg)
    {
        return std::real(arg) < 0.0 && std::abs(std::imag(arg)) <= -1.0e-9 * std::real(arg);
    }

    template <typename dtype>
    struct MaterialCols
    {
        arma::uword n_mtl = 0;         // Number of materials
        const dtype *fRef = nullptr;   // Reference frequency, GHz
        const dtype *a = nullptr;      // εr at fRef
        const dtype *b = nullptr;      // Frequency exponent for εr
        const dtype *c = nullptr;      // σ at fRef, S/m
        const dtype *d = nullptr;      // Frequency exponent for σ
        const dtype *e = nullptr;      // μr at fRef
        const dtype *f = nullptr;      // Frequency exponent for μr
        const dtype *g = nullptr;      // σμ (magnetic loss) at fRef
        const dtype *h = nullptr;      // Frequency exponent for σμ
        const dtype *thk = nullptr;    // Material thickness, m
        const dtype *att = nullptr;    // Penetration loss at fRef, dB
        const dtype *attB = nullptr;   // Frequency exponent for att
        const dtype *alpha = nullptr;  // In-medium absorption at fRef, dB/m
        const dtype *alphaB = nullptr; // Frequency exponent for alpha
        const dtype *m = nullptr;      // Mass-law transmission slope, dB/decade
        const dtype *resF = nullptr;   // Permittivity resonance frequency, GHz
        const dtype *resQ = nullptr;   // Permittivity resonance quality factor
        const dtype *resS = nullptr;   // Permittivity resonance strength
        const dtype *coiF = nullptr;   // Coincidence frequency, GHz
        const dtype *coiQ = nullptr;   // Coincidence quality factor
        const dtype *coiA = nullptr;   // Coincidence loss amplitude, dB
        const dtype *tf = nullptr;     // Transmission factor at fRef
        const dtype *tfB = nullptr;    // Frequency exponent for tf

        MaterialCols() = default; // All pointers stay nullptr

        MaterialCols(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop)
        {
            // Validate: all non-empty columns must have the same length
            n_mtl = 0;
            bool seen = false;
            for (const auto &kv : mtl_prop)
            {
                if (kv.second.empty())
                    continue;
                arma::uword len = (arma::uword)kv.second.size();
                if (!seen)
                {
                    n_mtl = len;
                    seen = true;
                }
                else if (len != n_mtl)
                    throw std::invalid_argument("Material property column '" + kv.first + "' has length " +
                                                std::to_string(len) + ", expected " + std::to_string(n_mtl) +
                                                " (all columns must have the same number of materials).");
            }

            // Lambda to resolve a column to its pointer (or nullptr if absent/empty)
            auto resolve = [&mtl_prop](const std::string &key) -> const dtype *
            {
                auto it = mtl_prop.find(key);
                return (it == mtl_prop.end() || it->second.empty()) ? nullptr : it->second.data();
            };

            // Assign columns
            fRef = resolve("fRef");
            a = resolve("a");
            b = resolve("b");
            c = resolve("c");
            d = resolve("d");
            e = resolve("e");
            f = resolve("f");
            g = resolve("g");
            h = resolve("h");
            thk = resolve("thk");
            att = resolve("att");
            attB = resolve("attB");
            alpha = resolve("alpha");
            alphaB = resolve("alphaB");
            m = resolve("m");
            resF = resolve("resF");
            resQ = resolve("resQ");
            resS = resolve("resS");
            coiF = resolve("coiF");
            coiQ = resolve("coiQ");
            coiA = resolve("coiA");
            tf = resolve("tf");
            tfB = resolve("tfB");

            // Physical sanity: reject corrupt input rather than silently clamping it. Loss-like terms must
            // be non-negative (a negative would be gain); material constants must be positive. Reported
            // material index is 1-based to match the public material indexing.
            auto require = [this](const dtype *col, const char *name, bool strict_positive)
            {
                if (col == nullptr)
                    return;
                for (arma::uword i = 0; i < n_mtl; ++i)
                {
                    double v = (double)col[i];
                    if (strict_positive ? (v <= 0.0) : (v < 0.0))
                        throw std::invalid_argument(std::string("Material property '") + name + "' = " +
                                                    std::to_string(v) + " at material " + std::to_string(i + 1) +
                                                    (strict_positive ? " must be positive." : " must be non-negative."));
                }
            };

            require(fRef, "fRef", true);    // reference frequency > 0
            require(a, "a", true);          // relative permittivity > 0 (use >= 1 for a strict vacuum floor)
            require(e, "e", true);          // relative permeability > 0 (diamagnets < 1 allowed)
            require(c, "c", false);         // conductivity >= 0
            require(g, "g", false);         // magnetic loss >= 0
            require(thk, "thk", false);     // thickness >= 0
            require(att, "att", false);     // penetration loss >= 0
            require(alpha, "alpha", false); // in-medium absorption >= 0
            require(m, "m", false);         // mass-law slope >= 0
            require(resF, "resF", false);   // resonance frequency >= 0 (0 disables)
            require(resQ, "resQ", false);   // resonance Q >= 0 (0 disables)
            require(coiF, "coiF", false);   // coincidence frequency >= 0 (0 disables)
            require(coiQ, "coiQ", false);   // coincidence Q >= 0
        }

        // Flag for a "thin-sheet" material
        bool is_sheet(arma::uword idx) const
        {
            if (idx > n_mtl)
                throw std::out_of_range("Material index " + std::to_string(idx) +
                                        " out of range [0, " + std::to_string(n_mtl) + "]");

            if (thk == nullptr)
                return false;
            if (idx == 0) // Not in table, default
                return false;
            if (thk[idx - 1] == (dtype)0.0)
                return false;
            return true;
        }
    };

    // Material struct
    struct Material
    {
        double fRef = 1.0;   // Reference frequency, GHz
        double a = 1.0;      // εr at fRef
        double b = 0.0;      // Frequency exponent for εr
        double c = 0.0;      // σ at fRef, S/m
        double d = 0.0;      // Frequency exponent for σ
        double e = 1.0;      // μr at fRef
        double f = 0.0;      // Frequency exponent for μr
        double g = 0.0;      // σμ (magnetic loss) at fRef
        double h = 0.0;      // Frequency exponent for σμ
        double thk = 0.0;    // Thickness, m
        double att = 0.0;    // Penetration loss at fRef, dB
        double attB = 0.0;   // Frequency exponent for att
        double alpha = 0.0;  // In-medium absorption at fRef, dB/m
        double alphaB = 0.0; // Frequency exponent for alpha
        double m = 0.0;      // Mass-law transmission slope, dB/decade
        double resF = 0.0;   // Permittivity resonance frequency, GHz
        double resQ = 0.0;   // Permittivity resonance quality factor
        double resS = 0.0;   // Permittivity resonance strength
        double coiF = 0.0;   // Coincidence frequency, GHz
        double coiQ = 0.0;   // Coincidence quality factor
        double coiA = 0.0;   // Coincidence loss amplitude, dB
        double tfR = 0.0;    // Transmission factor at fRef
        double tfB = 0.0;    // Frequency exponent for tf

        Material() = default; // All pointers stay nullptr

        template <typename dtype>
        Material(const MaterialCols<dtype> &cols, arma::uword idx = 0) // 1-based index, 0 = default (no material)
        {
            if (idx > cols.n_mtl)
                throw std::out_of_range("Material index " + std::to_string(idx) +
                                        " out of range [0, " + std::to_string(cols.n_mtl) + "]");

            if (idx == 0) // no material -> keep defaults (air / vacuum)
                return;

            arma::uword i = idx - 1; // 1-based index -> 0-based column position

            fRef = cols.fRef ? (double)cols.fRef[i] : fRef;
            a = cols.a ? (double)cols.a[i] : a;
            b = cols.b ? (double)cols.b[i] : b;
            c = cols.c ? (double)cols.c[i] : c;
            d = cols.d ? (double)cols.d[i] : d;
            e = cols.e ? (double)cols.e[i] : e;
            f = cols.f ? (double)cols.f[i] : f;
            g = cols.g ? (double)cols.g[i] : g;
            h = cols.h ? (double)cols.h[i] : h;
            thk = cols.thk ? (double)cols.thk[i] : thk;
            att = cols.att ? (double)cols.att[i] : att;
            attB = cols.attB ? (double)cols.attB[i] : attB;
            alpha = cols.alpha ? (double)cols.alpha[i] : alpha;
            alphaB = cols.alphaB ? (double)cols.alphaB[i] : alphaB;
            m = cols.m ? (double)cols.m[i] : m;
            resF = cols.resF ? (double)cols.resF[i] : resF;
            resQ = cols.resQ ? (double)cols.resQ[i] : resQ;
            resS = cols.resS ? (double)cols.resS[i] : resS;
            coiF = cols.coiF ? (double)cols.coiF[i] : coiF;
            coiQ = cols.coiQ ? (double)cols.coiQ[i] : coiQ;
            coiA = cols.coiA ? (double)cols.coiA[i] : coiA;
            tfR = cols.tf ? (double)cols.tf[i] : tfR;
            tfB = cols.tfB ? (double)cols.tfB[i] : tfB;
        }

        // Relative permittivity
        std::complex<double> eta(double fGHz = 1.0) const
        {
            double f_rel = fGHz / fRef;
            double eta_r = a * std::pow(f_rel, b);
            double sigma = c * std::pow(f_rel, d);
            double eta_i = -17.98 * sigma / fGHz;
            return std::complex<double>(eta_r, eta_i);
        }

        // Relative permeability
        std::complex<double> mu(double fGHz = 1.0) const
        {
            double f_rel = fGHz / fRef;
            double mu_r = e * std::pow(f_rel, f);
            double sigma_m = g * std::pow(f_rel, h);
            return std::complex<double>(mu_r, -17.98 * sigma_m / fGHz);
        }

        // Permittivity resonance (acoustic): complex Lorentz pole added to the interface (Fresnel)
        std::complex<double> eta_resonance(double fGHz = 1.0) const
        {
            if (resF <= 0.0 || resQ <= 0.0 || resS == 0.0)
                return std::complex<double>(0.0, 0.0);
            double resF2 = resF * resF;
            std::complex<double> denom(resF2 - fGHz * fGHz, (resF / resQ) * fGHz);
            return (resS * resF2) / denom;
        }

        // In-medium gain, linear
        double medium_gain(double dist_refract,           // In-medium distance (refracted path)
                           double fGHz = 1.0,             // Frequency
                           double abs_cos_theta_t = 1.0,  // Cosine of incidence angle @ FBS/VBS (refracted path)
                           double dist_geo = 0.0,         // Geometric distance (0.0 = dist_refract)
                           double abs_cos_theta = 0.0,    // Cosine of incidence angle @ orig (only used when dist_geo > 0)
                           double mass_min_path = 0.0015, // Shortest geometric path that still carries a mass-law term
                           double mass_normal = -1.0,     // Surface-mass surrogate with the angle removed (< 0 = dist_geo * abs_cos_theta)
                           double mass_cos = -1.0,        // Angle factor for the mass-law term (< 0 = abs_cos_theta)
                           double mass_bonded = 0.0) const // Summed X0 of the bonded mass-law bodies already crossed
        {
            if (dist_geo == 0.0) // Identity fallback
                dist_geo = dist_refract, abs_cos_theta = abs_cos_theta_t;

            std::complex<double> eta_val = eta(fGHz) * mu(fGHz);
            double er = std::real(eta_val);
            double tan_delta = std::imag(eta_val) / er;
            double cos_delta = 1.0 / std::sqrt(1.0 + tan_delta * tan_delta);
            double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
            Delta = std::sqrt(Delta) * 0.0477135 / (fGHz * std::sqrt(er));
            double loss = dist_refract * 8.686 / Delta;
            loss += dist_refract * alpha * std::pow(fGHz / fRef, alphaB);

            // The mass-law floor guards traced solids, where a sub-millimeter path is a meshing
            // artifact rather than a real traversal. A thin sheet is a real traversal at any
            // thickness, so the sheet path passes 0.0 and keeps the term.
            //
            // The panel's transfer impedance and the fluid's wave admittance enter the transmission
            // coefficient as one product, x = Z_panel * cos(theta) / (2 z0), so the term is a
            // function of that single argument and carries two slopes that are not the same
            // quantity. The panel side carries the fitted slope m, in dB per decade of frequency
            // and of surface mass. The angle side carries 20 dB per decade of cos(theta), which
            // comes from the fluid admittance rather than from the panel and therefore does not
            // follow m. They were tied together while the angle sat inside the same logarithm as
            // the frequency; mass_normal and mass_cos separate them.
            //
            // mass_normal is the surface-mass surrogate with the angle removed and mass_cos is the
            // angle factor. A caller supplying neither gets dist_geo * abs_cos_theta as the
            // surrogate and abs_cos_theta as the factor, whose product is the dist_geo * cos^2 term
            // this was written in before: across a slab traversal dist_geo is the normal thickness
            // divided by the cosine, so the surrogate is the normal thickness.
            //
            // mass_A is the normal-incidence term, unchanged in form and in value, including its
            // clip at an argument of one. The angle carries the mass law's own shape,
            // log10(1 + x^2), normalized so that the two endpoints are exact: mass_A at normal
            // incidence and zero at tangency. The normalization matters because mass_A is a fitted
            // lumped quantity rather than one derived from a density and a thickness, so the
            // material table owns the normal-incidence value and the angle law must not move it.
            // For any panel with meaningful loss the normalization factor is within 10*eps/mass_A
            // of ten - 5e-4 relative at 25 dB - so this is the exact single-panel mass law wherever
            // the term is worth anything.
            //
            // The angle dependence has no cutoff. It tapers to zero and is never negative, and its
            // knee sits at x = 1, fixed by m, the frequency and the surface mass, and by no
            // geometry.
            if (m > 0.0 && dist_geo > mass_min_path)
            {
                double mass_n = (mass_normal >= 0.0) ? mass_normal : dist_geo * abs_cos_theta;
                double mass_c = (mass_cos >= 0.0) ? mass_cos : abs_cos_theta;
                double mass_arg = (fGHz / fRef) * mass_n;
                double mass_A = (mass_arg > 0.0) ? m * std::log10(mass_arg) : 0.0;

                // A body bonded to mass-law bodies the wave has already crossed is not a second
                // panel. With no air gap the bodies move with one velocity, so their transfer
                // impedances are in series and what adds is X0, not the term in dB. The plate is
                // one body with the summed X0, and the charging telescopes: the first body crossed
                // paid its own term, and this one pays what the plate is worth above what was
                // paid, TL(X_run + X0) - TL(X_run). Each body contributes its X0 unclipped, so the
                // hard zero sits on the sum. With nothing crossed before, the branch below runs
                // exactly as it always did.
                if (mass_bonded > 0.0)
                {
                    if (mass_arg > 0.0)
                        loss += mass_TL(mass_bonded + std::pow(mass_arg, 0.05 * m), mass_c) - mass_TL(mass_bonded, mass_c);
                }
                else if (mass_A > 0.0 && mass_c > 0.0)
                {
                    if (mass_c >= 1.0) // normal incidence, and the guard against a cosine over one
                        loss += mass_A;
                    else
                    {
                        double X0 = std::pow(mass_arg, 0.05 * m); // = 10^(mass_A/20)
                        double x = X0 * mass_c;
                        loss += mass_A * std::log10(1.0 + x * x) / std::log10(1.0 + X0 * X0);
                    }
                }
            }
            return std::pow(10.0, -0.1 * loss);
        }

        // Impedance ratio of this body with the angle removed, X0 = ((f / fRef) M)^(m / 20), from its
        // surface-mass surrogate M. It is 10^(A / 20) of the normal-incidence term A wherever that
        // term is positive, and is not clipped below it. Zero for a material without the column.
        double mass_X0(double mass_normal, double fGHz = 1.0) const
        {
            if (!(m > 0.0))
                return 0.0;
            double mass_arg = (fGHz / fRef) * mass_normal;
            return (mass_arg > 0.0) ? std::pow(mass_arg, 0.05 * m) : 0.0;
        }

        // Mass term in dB of one plate with impedance ratio X0 at angle factor mass_c: the
        // single-body law of medium_gain written in X0, zero below an X0 of one.
        static double mass_TL(double X0, double mass_c)
        {
            if (!(X0 > 1.0) || !(mass_c > 0.0))
                return 0.0;
            double A = 20.0 * std::log10(X0);
            if (mass_c >= 1.0)
                return A;
            double x = X0 * mass_c;
            return A * std::log10(1.0 + x * x) / std::log10(1.0 + X0 * X0);
        }

        // Linear gain of the non-conductive absorption only: the alpha column, without the
        // conductivity loss that a complex refractive index already carries. A caller that takes the
        // conductivity and the propagation phase from one complex expression, exp(-j k0 n L), must not
        // draw the same loss from medium_gain a second time; it takes this instead.
        double excess_gain(double dist, double fGHz = 1.0) const
        {
            double loss = dist * alpha * std::pow(fGHz / fRef, alphaB);
            return std::pow(10.0, -0.1 * loss);
        }

        // Per-entry interface gain, linear
        double interface_gain(double fGHz = 1.0) const
        {
            double loss = att * std::pow(fGHz / fRef, attB);
            if (coiF > 0.0 && coiA != 0.0)
            {
                double x = coiQ * (fGHz - coiF) / coiF;
                loss += coiA / (1.0 + x * x);
            }
            if (loss < 0.0)
                return 1.0;
            return std::pow(10.0, -0.1 * loss);
        }

        // Transmission factor at fGHz, clamped to [-1, 1]
        double tf(double fGHz = 1.0) const
        {
            double v = tfR * std::pow(fGHz / fRef, tfB);
            return (v < -1.0) ? -1.0 : ((v > 1.0) ? 1.0 : v);
        }

        // Redistribute physical reflection energy R0 in [0,1] by tf in [-1,1], keeping refl + trans = 1
        double apply_tf(double R0, double fGHz = 1.0) const
        {
            double tf_val = tf(fGHz);
            R0 = (R0 < 0.0) ? 0.0 : ((R0 > 1.0) ? 1.0 : R0); // guard against resonance overshoot
            return (tf_val >= 0.0) ? R0 * (1.0 - tf_val) : R0 + (1.0 - R0) * (-tf_val);
        }

        // Combined transmission-factor reflection for a two-medium interface (symmetric in the two media).
        // tf+ = max(tf,0) leaks reflection energy into transmission; tf- = max(-tf,0) forces reflection.
        // Reduces to apply_tf at an air boundary (other side tf = 0); stays in [0,1]; tf = -1 on either
        // face gives R_eff = 1.
        double apply_tf_pair(const Material &other, double R0, double fGHz = 1.0) const
        {
            double tfX = tf(fGHz), tfY = other.tf(fGHz);
            double tfXp = (tfX > 0.0) ? tfX : 0.0, tfXm = (tfX < 0.0) ? -tfX : 0.0;
            double tfYp = (tfY > 0.0) ? tfY : 0.0, tfYm = (tfY < 0.0) ? -tfY : 0.0;
            R0 = (R0 < 0.0) ? 0.0 : ((R0 > 1.0) ? 1.0 : R0);
            double R_leak = R0 * (1.0 - tfXp) * (1.0 - tfYp);
            double tfm = (tfXm > tfYm) ? tfXm : tfYm;
            return R_leak + (1.0 - R_leak) * tfm;
        }

        // Check whether two materials are the same
        bool same_as(const Material &other) const
        {
            return fRef == other.fRef &&
                   a == other.a &&
                   b == other.b &&
                   c == other.c &&
                   d == other.d &&
                   e == other.e &&
                   f == other.f &&
                   g == other.g &&
                   h == other.h &&
                   thk == other.thk &&
                   att == other.att &&
                   attB == other.attB &&
                   alpha == other.alpha &&
                   alphaB == other.alphaB &&
                   m == other.m &&
                   resF == other.resF &&
                   resQ == other.resQ &&
                   resS == other.resS &&
                   coiF == other.coiF &&
                   coiQ == other.coiQ &&
                   coiA == other.coiA &&
                   tfR == other.tfR &&
                   tfB == other.tfB;
        }

        // Per-channel normal-flux obliquity, returned as an amplitude factor.
        //
        // The time-averaged normal Poynting component of a plane wave is proportional to
        // |E|^2 Re(z c) for TE and to |E|^2 Re(conj(z) c) for TM, with z = sqrt(eps/mu). The energy
        // crossing the boundary is therefore Re(z2 c2)/Re(z1 c1) |t_TE|^2 in one channel and
        // Re(conj(z2) c2)/Re(conj(z1) c1) |t_TM|^2 in the other, and the two obliquity factors are
        // not the same once z2 is complex.
        //
        // Both cosines are complex here. For a real ct1 the two denominators coincide at
        // Re(z1)*ct1, since Re(conj(z1) ct1) = Re(z1 ct1) = Re(z1)*ct1, so an interface evaluated
        // from a real incidence angle is unchanged by the general form. They separate when the
        // incidence side is itself a lossy medium entered at an angle, which is the case at the far
        // face of a traced solid: there the propagation cosine recovered from the traced direction
        // is complex, and the incidence-side flux the transmitted port is measured against is
        // Re(z1 ct1) per channel rather than Re(z1)*|ct1|.
        //
        // Writing it this way is what makes a chain of crossings telescope. Two consecutive
        // crossings that share a medium contribute Re(z2 c2)/Re(z1 c1) and Re(z3 c3)/Re(z2 c2); the
        // shared flux cancels as a ratio of one and the same real number, leaving
        // Re(z3 c3)/Re(z1 c1), provided both crossings are evaluated at the same cosine in the
        // shared medium.
        //
        // For a lossless incidence medium this makes the per-channel transmittance identically
        // 1 - R_X, at any angle and for any complex second medium, so an air-side crossing keeps the
        // total it reported before. For a lossy incidence medium the two quantities differ, because
        // the incident and reflected waves interfere and the split into two independent fluxes is
        // not exact there; the flux form is the one that composes across a chain of crossings.
        //
        // Past the critical angle Re(z2 c2) vanishes for a lossless second medium, so the
        // transmitted port closes on its own without the magnitude test that declares TIR.
        //
        // The scalar branch uses the TE form: z is the acoustic admittance 1/(rho c) and the normal
        // intensity is |p|^2 Re(z c).
        static void flux_amp(const std::complex<double> &z1, const std::complex<double> &z2,
                             const std::complex<double> &ct1, const std::complex<double> &ct2,
                             double &s_te, double &s_tm)
        {
            double den_te = std::real(z1 * ct1); // incidence-side normal flux, per channel
            double den_tm = std::real(std::conj(z1) * ct1);
            double n_te = std::real(z2 * ct2);
            double n_tm = std::real(std::conj(z2) * ct2);

            // A non-positive denominator is exact tangency or a non-propagating incidence-side
            // wave: there is no incident flux to convert and the port closes.
            s_te = (den_te > 1.0e-300 && n_te > 0.0) ? std::sqrt(n_te / den_te) : 0.0;
            s_tm = (den_tm > 1.0e-300 && n_tm > 0.0) ? std::sqrt(n_tm / den_tm) : 0.0;
        }

        // Medium-medium interaction
        // 'this' is the medium the ray travels in (incidence side, medium 1); 'other' is the medium it enters into / reflects off (medium 2).
        // Computes the ITU-R P.2040-1 interface coefficients and returns the interface power gain.
        // For transmission/refraction (1/2/4/5), the returned gain and cTE/cTM include the entered medium's lumped
        // interface_gain (att + coincidence); reflection (0/3) does not. 0.5*(|cTE|^2 + |cTM|^2) == gain in all cases.
        //
        // Energy convention. A transmitted coefficient carries the per-channel normal-flux amplitude,
        // the raw Fresnel amplitude times sqrt of its own obliquity factor; see flux_amp above. The
        // undeviated and refracting types therefore report the same energy for the same interface,
        // which is what they owe once the outgoing ray tube carries no power of its own. Reflection
        // stays in one medium, where the obliquity factor is identically 1, and is unchanged.
        //
        // Incidence cosine. `theta` is a real angle and is what a face hit supplies. An interface
        // met from inside a traced medium has no real incidence angle to offer: the medium's
        // propagation cosine is complex there, and the tracer's direction carries only its real
        // projection. Such a caller passes the recovered value through cos_theta_in and `theta` is
        // then unused. Everything below is written in that cosine and is analytic in it, so the two
        // paths are one expression and a real cos_theta_in reproduces the angle path exactly.
        double interact_with(const Material &other,                              // Material that the path enters into / reflects of
                             int interaction_type,                               // 0 = EM reflect, 1 = EM transmit, 2 = EM refract, 3 = scalar reflect, 4 = scalar transmit, 5 = scalar refract
                             double theta,                                       // Incidence angle (ignored when cos_theta_in is given)
                             double fGHz,                                        // Frequency
                             std::complex<double> *cTE = nullptr,                // Out: E-field coefficient, R for reflection (0/3), T for transmission/refraction (1/2/4)
                             std::complex<double> *cTM = nullptr,                // Out: M-field coefficient
                             std::complex<double> *cos_theta2 = nullptr,         // Out: Refraction cosine (type-2 direction)
                             std::complex<double> *eta1_div_eta2 = nullptr,      // Out: eta1/eta2
                             double *Snell_ratio = nullptr,                      // Out: sqrt|eta1*mu1 / eta2*mu2| (Snell ratio, type-2 direction)
                             bool *total_reflection = nullptr,                   // Out: Total reflection indicator
                             bool force_tir = false,                             // Switch to force total internal reflection
                             const std::complex<double> *cos_theta_in = nullptr, // Incidence cosine, complex; overrides theta
                             bool undeviated = false) const                      // The wave is held at an angle it was not refracted to
        {
            // Interface geometry, shared by every interaction type.
            //
            // The real-angle path forms sin^2 as sqrt(1 - c^2) squared, which is not the same double
            // as 1 - c^2, and keeps its own total-reflection test. Both are reproduced here rather
            // than folded into the complex form: at a matched interface the difference is one ulp in
            // ct2, which is enough to lift the reflectance off exactly zero and send a tf-forced
            // reflection down the rescale branch instead of the flat one, flipping the sign of a
            // coefficient whose power is unchanged. Nothing on the angle path may move.
            std::complex<double> ct1, sin2_theta;
            double sin_theta_real = -1.0; // >= 0 on the angle path only, and its TIR test
            if (cos_theta_in)
            {
                ct1 = *cos_theta_in;
                sin2_theta = 1.0 - ct1 * ct1;
            }
            else
            {
                double abs_cos_theta = std::abs(std::cos(theta + 1.570796326794897)); // |OF . N| convention
                abs_cos_theta = (abs_cos_theta > 1.0) ? 1.0 : abs_cos_theta;
                sin_theta_real = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta);
                ct1 = std::complex<double>(abs_cos_theta, 0.0);
                sin2_theta = std::complex<double>(sin_theta_real * sin_theta_real, 0.0);
            }

            std::complex<double> eta1 = eta(fGHz) + eta_resonance(fGHz); // incidence medium (this)
            std::complex<double> eta2 = other.eta(fGHz) + other.eta_resonance(fGHz);
            std::complex<double> mu1 = mu(fGHz);
            std::complex<double> mu2 = other.mu(fGHz);

            std::complex<double> eta1_d_eta2 = (eta1 * mu1) / (eta2 * mu2);
            double snell = std::sqrt(std::abs(eta1_d_eta2));

            // Same value either way; written with the original association on the angle path so
            // that path stays bit-identical. Reassociating it moved coefficients by up to 6e-14
            // relative, which is above the 1e-12 the per-channel split is asserted to.
            std::complex<double> ct2_arg = (sin_theta_real >= 0.0)
                                               ? (1.0 - eta1_d_eta2 * sin_theta_real * sin_theta_real)
                                               : (1.0 - eta1_d_eta2 * sin2_theta);
            if (qd_evanescent(ct2_arg))
                ct2_arg = std::complex<double>(std::real(ct2_arg), -0.0); // TIR branch cut: decaying evanescent wave
            std::complex<double> ct2 = std::sqrt(ct2_arg);
            std::complex<double> z1 = std::sqrt(eta1 / mu1); // admittances sqrt(eps/mu)
            std::complex<double> z2 = std::sqrt(eta2 / mu2);

            // Total reflection, written in the transverse term the refraction root is built from.
            // For a real angle this is snell * sin(theta) >= 1, unchanged. For a recovered complex
            // cosine the same product is the transverse term carried in from the medium the wave
            // entered through - Snell is transitive, so eta1_d_eta2 * sin2_theta equals
            // (eta_entry mu_entry / eta2 mu2) sin^2(theta_entry) exactly - so the verdict is the one
            // the outer pair of media gives, which is the only pair that can create or remove total
            // reflection. The two forms are the same statement; they are written separately only so
            // the angle path keeps its own rounding.
            //
            // It is a critical angle only where the second medium is lossless. A lossy one always
            // admits a transmitted wave, with a complex cosine, and flux_amp closes no port on it,
            // so there the test is a statement about a magnitude and not about the wave: a porous
            // absorber, rarer than air, turned into a mirror from 37 deg. Decided on the medium
            // and not on a value, so it is one verdict for every frequency of a run.
            bool tir = force_tir || (std::imag(eta2 * mu2) == 0.0 &&
                                     ((sin_theta_real >= 0.0) ? (snell * sin_theta_real >= 1.0)
                                                              : (std::abs(eta1_d_eta2 * sin2_theta) >= 1.0)));

            std::complex<double> coeff_TE, coeff_TM;
            double gain;

            // A face is not judged from inside a mass-law medium by a wave that is held undeviated.
            //
            // A material with m > 0 is calibrated for surface impedance only, so its eta * mu is not
            // a propagation index, and the critical angle of a face seen from inside it is formed
            // from exactly that index. A tf leak keeps its direction, so the reverse of a leak that
            // crossed this face at an angle is the crossing from the other side at the same angle.
            // Judged from here that crossing is total reflection past the critical angle and passes
            // only the flat leak; judged from there it never is and is an ordinary crossing. Same
            // face, same media, same angle, so the two are made one number: the wave is charged what
            // the other direction charges, with the interface gain of the medium it enters. That
            // crossing arrives in this medium's neighbour at the angle the wave keeps, so it leaves
            // undeviated and the verdict handed back is the one that says so.
            //
            // Only towards the lower index, where this side has a critical angle at all; the other
            // direction is the ordinary crossing and is untouched. Only for a wave that was not
            // refracted in: one the tracer bent arrives at a Snell angle, where the crossing from
            // this side is already the reverse of its entry.
            //
            // The reflect port follows the same rule, or the two ports of one face do not add up:
            // judged from inside it is a mirror past the critical angle while the transmit port
            // passes what the other direction passes. It is the reflection the other direction
            // sees at this cosine with its sign reversed, which is Stokes' relation for one face
            // read from its two sides. It stays in this medium, so it carries no interface gain,
            // and the verdict handed back is the ordinary one: a reflection has no bend to hold.
            bool trans_class = interaction_type == 1 || interaction_type == 2 || interaction_type == 4 || interaction_type == 5;
            if (undeviated && m > 0.0 && !trans_class && std::abs(eta1_d_eta2) > 1.0)
            {
                gain = other.interact_with(*this, interaction_type, theta, fGHz, &coeff_TE, &coeff_TM, nullptr, nullptr,
                                           nullptr, nullptr, false, cos_theta_in, false);
                if (cTE)
                    *cTE = -coeff_TE;
                if (cTM)
                    *cTM = -coeff_TM;
                if (cos_theta2)
                    *cos_theta2 = ct2;
                if (eta1_div_eta2)
                    *eta1_div_eta2 = eta1_d_eta2;
                if (Snell_ratio)
                    *Snell_ratio = snell;
                if (total_reflection)
                    *total_reflection = tir;
                return gain;
            }
            if (undeviated && m > 0.0 && trans_class && std::abs(eta1_d_eta2) > 1.0)
            {
                gain = other.interact_with(*this, interaction_type, theta, fGHz, &coeff_TE, &coeff_TM, nullptr, nullptr,
                                           nullptr, nullptr, false, cos_theta_in, false);
                double s = other.interface_gain(fGHz) / interface_gain(fGHz); // the gain of the medium entered
                gain *= s;
                coeff_TE *= std::sqrt(s), coeff_TM *= std::sqrt(s);

                if (cTE)
                    *cTE = coeff_TE;
                if (cTM)
                    *cTM = coeff_TM;
                if (cos_theta2)
                    *cos_theta2 = ct2;
                if (eta1_div_eta2)
                    *eta1_div_eta2 = eta1_d_eta2;
                if (Snell_ratio)
                    *Snell_ratio = snell;
                if (total_reflection)
                    *total_reflection = true;
                return gain;
            }

            if (interaction_type < 3) // EM: types 0 (reflection), 1 (transmission), 2 (refraction)
            {
                // Fresnel reflection (TE/TM), ITU-R P.2040-1 eq. (31); under TIR the interface is a perfect mirror
                std::complex<double> R_TE = (z1 * ct1 - z2 * ct2) / (z1 * ct1 + z2 * ct2);
                std::complex<double> R_TM = (z2 * ct1 - z1 * ct2) / (z2 * ct1 + z1 * ct2);
                double reflectance = tir ? 1.0 : 0.5 * (std::norm(R_TE) + std::norm(R_TM));

                // reflectance is the TE/TM average, which is what the returned gain is defined to be
                // for a reflection. It does not distort the per-channel split there: the coefficients
                // below are rescaled by one factor common to both channels, so their ratio is the raw
                // Fresnel one throughout, and at tf = 0 that factor is identically 1. tf redistributes
                // energy between the reflected and transmitted ports and is polarization-agnostic by
                // design; if it is ever made polarization-dependent, this average is the first thing
                // that has to change. The transmission branch below no longer rescales by an averaged
                // quantity at all - each channel carries its own obliquity factor - and reads the
                // average only through tf.

                if (interaction_type == 0) // EM reflection: tf-adjusted reflectance
                {
                    gain = apply_tf_pair(other, reflectance, fGHz);
                    if (reflectance > 0.0) // rescale the Fresnel R coefficients to carry tf
                    {
                        double s = std::sqrt(gain / reflectance);
                        coeff_TE = R_TE * s, coeff_TM = R_TM * s;
                    }
                    else // tf < 0 created reflection at a zero-reflectance interface -> flat coefficients
                        coeff_TE = coeff_TM = std::complex<double>(std::sqrt(gain), 0.0);
                }
                else // EM transmission (1) or refraction (2): one branch, one convention
                {
                    double R_eff = apply_tf_pair(other, reflectance, fGHz);

                    if (tir) // no Fresnel forward port; any tf leak travels undeviated, flat coefficients
                    {
                        gain = 1.0 - R_eff;
                        coeff_TE = coeff_TM = std::complex<double>(std::sqrt(gain), 0.0);
                    }
                    else
                    {
                        // Fresnel transmission, ITU-R P.2040-1 eq. (32)
                        std::complex<double> T_TE = (2.0 * z1 * ct1) / (z1 * ct1 + z2 * ct2);
                        std::complex<double> T_TM = (2.0 * z1 * ct1) / (z2 * ct1 + z1 * ct2);

                        // tf redistributes between the ports and is not per-channel, so it enters as
                        // one scale on top of the per-channel flux amplitudes. It is 1 at tf = 0.
                        double tf_scale = (reflectance < 1.0) ? (1.0 - R_eff) / (1.0 - reflectance) : 0.0;
                        double s_te = 0.0, s_tm = 0.0;
                        flux_amp(z1, z2, ct1, ct2, s_te, s_tm);
                        double a = std::sqrt(tf_scale);
                        coeff_TE = T_TE * (s_te * a);
                        coeff_TM = T_TM * (s_tm * a);
                        gain = 0.5 * (std::norm(coeff_TE) + std::norm(coeff_TM));
                    }
                }
            }
            else // Scalar (acoustic): types 3 (reflection), 4 (transmission), 5 (refraction)
            {
                std::complex<double> R = (z1 * ct1 - z2 * ct2) / (z1 * ct1 + z2 * ct2);
                double reflectance = tir ? 1.0 : std::norm(R);
                double R_eff = apply_tf_pair(other, reflectance, fGHz);
                std::complex<double> T = 1.0 + R; // pressure transmission coefficient

                if (interaction_type == 3) // Scalar reflection: tf-adjusted reflectance
                {
                    gain = R_eff;
                    coeff_TE = std::polar(std::sqrt(gain), std::arg(R));
                }
                else if (tir) // Transmission (4) or refraction (5) with no forward port: flat tf leak
                {
                    gain = 1.0 - R_eff;
                    coeff_TE = std::polar(std::sqrt(gain), std::arg(T));
                }
                else // Scalar transmission (4) and refraction (5): one branch, one convention
                {
                    double tf_scale = (reflectance < 1.0) ? (1.0 - R_eff) / (1.0 - reflectance) : 0.0;
                    double s_te = 0.0, s_tm = 0.0;
                    flux_amp(z1, z2, ct1, ct2, s_te, s_tm); // s_tm unused, one channel
                    coeff_TE = T * (s_te * std::sqrt(tf_scale));
                    gain = std::norm(coeff_TE);
                }
                coeff_TM = coeff_TE;
            }

            // Transmission-class interactions (1, 2, 4, 5) cross a thin interface: fold in the interface gain
            if (interaction_type == 1 || interaction_type == 2 || interaction_type == 4 || interaction_type == 5)
            {
                double ig = other.interface_gain(fGHz);
                gain *= ig;
                double s = std::sqrt(ig);
                coeff_TE *= s;
                coeff_TM *= s;
            }

            // A wave held undeviated in a mass-law medium keeps its direction at every face. The
            // medium is a lumped plate with no wave inside, and this face is charged the same from
            // both sides at one angle, so the wave has that angle on both sides. Towards the lower
            // index the verdict above says so. Towards the higher index the crossing is the
            // ordinary one and its Snell ratio is formed from two permittivities that are
            // impedance mappings, so the geometry handed back is that of no bend: a Snell ratio of
            // one and the incidence cosine, from which every caller reproduces the direction it
            // came in with. The coefficient and the verdict are the ordinary ones.
            if (undeviated && m > 0.0 && trans_class)
                ct2 = ct1, eta1_d_eta2 = 1.0, snell = 1.0;

            if (cTE)
                *cTE = coeff_TE;
            if (cTM)
                *cTM = coeff_TM;
            if (cos_theta2)
                *cos_theta2 = ct2;
            if (eta1_div_eta2)
                *eta1_div_eta2 = eta1_d_eta2;
            if (Snell_ratio)
                *Snell_ratio = snell;
            if (total_reflection)
                *total_reflection = tir;

            return gain;
        }
    };

    // Mirror reflection direction: d = u - 2*c*n, c = clamp(u.n)
    inline void qd_reflect(double Ux, double Uy, double Uz,    // Incoming direction, normalized
                           double Nx, double Ny, double Nz,    // Plane normal vector, normalized
                           double &Dx, double &Dy, double &Dz) // Outgoing direction (normalized if U and N are normalized)
    {
        double c = Ux * Nx + Uy * Ny + Uz * Nz;
        c = (c < -1.0) ? -1.0 : (c > 1.0 ? 1.0 : c);
        double a = 2.0 * c;
        Dx = Ux - a * Nx, Dy = Uy - a * Ny, Dz = Uz - a * Nz;
    }

    // Relaunch offset: a few float ULP at the FBS coordinate magnitude (2^-23), so the nudge survives
    // a store in single precision and the relaunched ray starts off the face without re-hitting it
    inline double qd_relaunch_offset(double Fx, double Fy, double Fz)
    {
        double scale = std::max({std::abs(Fx), std::abs(Fy), std::abs(Fz), 1.0});
        return 8.0 * scale * 1.1920929e-7;
    }

    // Snell refraction direction: normalize(eta*u + (eta*cos_in - Re(cos_theta2))*n)
    inline void qd_refract(double Ux, double Uy, double Uz,    // Incoming direction, normalized
                           double Nx, double Ny, double Nz,    // Plane normal vector, normalized
                           double eta,                         // Snell ratio sqrt|eta1*mu1 / eta2*mu2|
                           double cos_in,                      // Cosine of angle between normal vector and incoming ray
                           std::complex<double> cos_theta2,    // Cosine of angle between normal vector and outgoing ray
                           double &Dx, double &Dy, double &Dz) // Outgoing direction (normalized)
    {
        double s = eta * cos_in - std::real(cos_theta2);
        double Rx = eta * Ux + s * Nx, Ry = eta * Uy + s * Ny, Rz = eta * Uz + s * Nz;
        double inv = 1.0 / std::sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
        Dx = Rx * inv, Dy = Ry * inv, Dz = Rz * inv;
    }

    // Builds incoming Q-basis from in, outgoing U-basis from out, writes the 8-element xprmat and
    // its power gain. is_scalar takes the single-coefficient path.
    inline void qd_polbasis(double Qx, double Qy, double Qz,
                            double Ux, double Uy, double Uz,
                            double Nx, double Ny, double Nz,
                            double amplitude, std::complex<double> cTE, std::complex<double> cTM,
                            bool is_scalar, double xprmat[8], double &out_gain)
    {
        double eTE_Re = std::real(cTE), eTE_Im = std::imag(cTE);
        double eTM_Re = std::real(cTM), eTM_Im = std::imag(cTM);

        if (is_scalar)
        {
            double coeff_Re = amplitude * eTE_Re, coeff_Im = amplitude * eTE_Im;
            xprmat[0] = coeff_Re, xprmat[1] = coeff_Im;
            xprmat[2] = 0.0, xprmat[3] = 0.0, xprmat[4] = 0.0;
            xprmat[5] = 0.0, xprmat[6] = 0.0, xprmat[7] = 0.0;
            out_gain = coeff_Re * coeff_Re + coeff_Im * coeff_Im;
            return;
        }

        double scl = 0.0;

        // Incoming path basis from (Qx,Qy,Qz)
        double eHx = -Qy + 3.0e-20, eHy = Qx, eHz = 0.0;
        scl = 1.0 / std::sqrt(eHx * eHx + eHy * eHy), eHx *= scl, eHy *= scl;
        double eVx = -Qz * eHy, eVy = Qz * eHx, eVz = Qx * eHy - Qy * eHx;
        double eQx = Qy * Nz - Qz * Ny + 3.0e-20, eQy = Qz * Nx - Qx * Nz, eQz = Qx * Ny - Qy * Nx;
        scl = 1.0 / std::sqrt(eQx * eQx + eQy * eQy + eQz * eQz), eQx *= scl, eQy *= scl, eQz *= scl;
        double ePx = eQy * Qz - eQz * Qy, ePy = eQz * Qx - eQx * Qz, ePz = eQx * Qy - eQy * Qx;

        bool do_base_transform = scl < 1.0e19;
        double Q1 = (do_base_transform) ? eVx * ePx + eVy * ePy + eVz * ePz : 1.0;
        double Q2 = (do_base_transform) ? eVx * eQx + eVy * eQy + eVz * eQz : 0.0;
        double Q3 = (do_base_transform) ? eHx * ePx + eHy * ePy + eHz * ePz : 0.0;
        double Q4 = (do_base_transform) ? eHx * eQx + eHy * eQy + eHz * eQz : 1.0;

        // Outgoing path basis from (Ux,Uy,Uz)
        eHx = -Uy + 3.0e-20, eHy = Ux, eHz = 0.0;
        scl = 1.0 / std::sqrt(eHx * eHx + eHy * eHy), eHx *= scl, eHy *= scl;
        eVx = -Uz * eHy, eVy = Uz * eHx, eVz = Ux * eHy - Uy * eHx;
        eQx = Uy * Nz - Uz * Ny + 3.0e-20, eQy = Uz * Nx - Ux * Nz, eQz = Ux * Ny - Uy * Nx;
        scl = 1.0 / std::sqrt(eQx * eQx + eQy * eQy + eQz * eQz), eQx *= scl, eQy *= scl, eQz *= scl;
        ePx = eQy * Uz - eQz * Uy, ePy = eQz * Ux - eQx * Uz, ePz = eQx * Uy - eQy * Ux;

        do_base_transform = scl < 1.0e19;
        double U1 = (do_base_transform) ? eVx * ePx + eVy * ePy + eVz * ePz : 1.0;
        double U2 = (do_base_transform) ? eVx * eQx + eVy * eQy + eVz * eQz : 0.0;
        double U3 = (do_base_transform) ? eHx * ePx + eHy * ePy + eHz * ePz : 0.0;
        double U4 = (do_base_transform) ? eHx * eQx + eHy * eQy + eHz * eQz : 1.0;

        double VV_Re = amplitude * (U1 * Q1 * eTM_Re + U3 * Q2 * eTE_Re),
               VV_Im = amplitude * (U1 * Q1 * eTM_Im + U3 * Q2 * eTE_Im),
               HV_Re = amplitude * (U2 * Q1 * eTM_Re + U4 * Q2 * eTE_Re),
               HV_Im = amplitude * (U2 * Q1 * eTM_Im + U4 * Q2 * eTE_Im),
               VH_Re = amplitude * (U1 * Q3 * eTM_Re + U3 * Q4 * eTE_Re),
               VH_Im = amplitude * (U1 * Q3 * eTM_Im + U3 * Q4 * eTE_Im),
               HH_Re = amplitude * (U2 * Q3 * eTM_Re + U4 * Q4 * eTE_Re),
               HH_Im = amplitude * (U2 * Q3 * eTM_Im + U4 * Q4 * eTE_Im);

        xprmat[0] = VV_Re, xprmat[1] = VV_Im;
        xprmat[2] = HV_Re, xprmat[3] = HV_Im;
        xprmat[4] = VH_Re, xprmat[5] = VH_Im;
        xprmat[6] = HH_Re, xprmat[7] = HH_Im;

        out_gain = 0.5 * (VV_Re * VV_Re + VV_Im * VV_Im +
                          HV_Re * HV_Re + HV_Im * HV_Im +
                          VH_Re * VH_Re + VH_Im * VH_Im +
                          HH_Re * HH_Re + HH_Im * HH_Im);
    }

    // One layer of a stratified stack: a material and the normal thickness it occupies.
    struct QdLayer
    {
        const Material *mtl = nullptr;
        double thk = 0.0;
    };

    // 2x2 characteristic-matrix resolve for an ordered stack of layers between two half-spaces.
    //
    // Why a matrix rather than another Airy sum. An Airy sum is a statement about one layer with a
    // half-space on either side. Two layers couple through reflections between non-adjacent
    // boundaries, and those do not factor into a product of per-layer series. The matrix carries the
    // whole stack in one pass and is exact for any layer count.
    //
    // The variables are the ones the rest of this file already uses. Per medium the admittance is
    // z = sqrt(eps/mu) and the index is n = sqrt(eps mu), and every cosine comes from the transverse
    // invariant eps mu (1 - c^2), which is conserved across every boundary and is fixed by the
    // incidence cosine in the near medium. Per polarization the tilted admittance is
    //
    //     eta_TE = z c        eta_TM = z / c
    //
    // which are the groupings Material::interact_with is written in: its TE reflection is
    // (eta1 - eta2)/(eta1 + eta2) exactly, and dividing its TM reflection through by c1 c2 turns it
    // into (eta2 - eta1)/(eta2 + eta1). So the library's TM reflection is minus the admittance-form
    // one and its TM transmission carries an extra c1/c2; both conversions are applied at the ends
    // below, and the c1/c2 telescopes over the stack because the intermediate cosines cancel.
    //
    // What propagates is the tangential field pair, not flux: tangential E and H are continuous
    // across every boundary while normal flux decays inside a lossy layer. So the product runs in
    // amplitudes and the conversion to per-channel normal flux is applied once, at the outer pair,
    // on the transmit port. Inside the product the intermediate admittances cancel identically,
    // which is the same statement qd_slab_resolve makes for one layer when it keeps the raw Fresnel
    // amplitudes.
    //
    // At one layer this reduces to the Airy pair produce mode returns, term for term, including the
    // Stokes sign on the near reflection, the flux conversion, the lumped interface gains and the
    // displaced host. Produce mode is deliberately NOT routed through here: its one-layer path is
    // asserted bit-identical to its pre-change form over 31104 rows, and the two forms are
    // algebraically equal but not bit-identical. The layered-stack fixture measures the two against
    // each other, so unifying them later is a checked step rather than a leap.
    //
    // Three material columns have no matrix expression, or none on the reflect port. None makes
    // this decline; a stack that carries one is resolved in the scattering form instead, and a
    // stack that carries none takes the matrix exactly as before.
    //
    //   tf        redistributes energy between the reflected and transmitted ports of a boundary
    //             after the Fresnel problem is solved, which is not a boundary admittance. The
    //             scattering form has the per-boundary r and t it rescales: each boundary a | b is
    //             the two-port (r, t_ab, t_ba, -r) in the amplitudes produce mode uses, rescaled
    //             under apply_tf_pair by the two media that form it - so a boundary shared by two
    //             sheets has no single owner - and each layer is the one-way factor phi. The
    //             product of the wave-transfer matrices B_1 P_1 B_2 ... B_{N+1} is evaluated by
    //             recursion from the far end,
    //                 G_j = r_j + t_j tb_j G_{j+1} phi^2 / (1 + r_j G_{j+1} phi^2)
    //                 T_j = t_j phi T_{j+1} / (1 + r_j G_{j+1} phi^2)
    //             which needs no inverse of a closed boundary (t = 0 at tf = -1) or of an opaque
    //             layer. At one layer it is produce mode's Airy pair term for term, and at tf = 0
    //             it is the matrix product in another basis.
    //   att       a layer's interface gain is an end factor on the transmit port, which the matrix
    //             carries. On the reflect port it is owed by what entered that layer and came
    //             back, once, and a matrix has no per-layer echo to charge; the recursion has,
    //             so a stack with a gain-bearing layer takes this form as well. The series it
    //             runs on is the one without the gain, so the transmit port is the matrix's.
    //   m > 0     a lumped acoustic surrogate whose in-layer index is not physical. Such a layer
    //             is not made coherent. It ends the coherent segment in front of it as a
    //             half-space of its own admittance, which is what its eta and mu are calibrated
    //             for, contributes one real amplitude factor, and starts the next segment as its
    //             incidence half-space. Nothing returns across it, so the reflect port is that of
    //             the first segment. With no coherent layer this is produce mode's mass-law route,
    //             and like that route it displaces no host thickness. Mass-law layers with
    //             no coherent layer between them are bonded and form one plate: their X0 add,
    //             the first pays its own term and each later one what the plate is worth above
    //             what was paid; see Material::medium_gain. mass_bonded is the X0 the wave has
    //             already crossed in a mass-law near medium.
    //
    // alpha is carried. It is dB per meter of PATH, which produce mode applies as a real one-way
    // multiplier on phi, and the layer phase is the logarithm of that same phi: the matrix is built
    // from cos and sin of delta, and phi = exp(-j delta). So the multiplier goes into the imaginary
    // part of delta, over the same path thk / |Re c| produce mode forms, and the one-layer reduction
    // then reproduces produce mode's alpha term for term. The tilted admittance does not read it,
    // since alpha is a propagation loss and not an interface quantity.
    //
    // Returns false when it declines. ci is the incidence cosine in mat_near, complex wherever that
    // medium is a traced solid, exactly as produce mode's ct_near_exact.
    inline bool qd_stack_resolve(const Material &mat_near,
                                 const QdLayer *layers, int n_layers,
                                 const Material &mat_far,
                                 std::complex<double> ci,
                                 double fGHz, bool is_scalar,
                                 std::complex<double> *T_te, std::complex<double> *T_tm,
                                 std::complex<double> *R_te, std::complex<double> *R_tm,
                                 bool disp_far = false, double disp_budget = -1.0,
                                 double mass_cos = -1.0,    // cosine the wave has in air, for a mass-law term, < 0 = |Re ci|
                                 double mass_bonded = 0.0,  // X0 of the bonded mass-law bodies crossed before the stack
                                 bool near_undeviated = false) // the wave in mat_near was not refracted in
    {
        if (n_layers < 1 || layers == nullptr)
            return false;
        bool scattering = mat_near.tf(fGHz) != 0.0 || mat_far.tf(fGHz) != 0.0;
        for (int i = 0; i < n_layers; ++i)
        {
            if (layers[i].mtl == nullptr || layers[i].thk < 0.0)
                return false;
            const Material &L = *layers[i].mtl;
            if (L.m > 0.0 || L.tf(fGHz) != 0.0 || L.interface_gain(fGHz) != 1.0)
                scattering = true;
        }

        // A stack of mass-law layers reached by a wave that is held undeviated in a mass-law
        // medium - two rigid sheets inside a rigid wall - is a run of later faces of that wall.
        // The wave is a leak, it keeps the incidence cosine at every face, and each face is the
        // crossing Material::interact_with forms for a held wave, with the interface gain of the
        // medium entered; each layer adds its mass term. That is what the same bodies do as
        // traced solids. The cosines of the segments below are Snell cosines of permittivities
        // that are impedance mappings, and do not describe this wave.
        //
        // Taken where every layer carries the mass column and a face leads towards the lower
        // index, which is where the undeviated rule has anything to say; decided on the media.
        if (near_undeviated && mat_near.m > 0.0)
        {
            auto W = [&](const Material &X) -> std::complex<double>
            { return (X.eta(fGHz) + X.eta_resonance(fGHz)) * X.mu(fGHz); };
            bool all_mass = true, to_lower = false;
            const Material *A = &mat_near;
            for (int i = 0; i < n_layers; ++i)
            {
                const Material &L = *layers[i].mtl;
                all_mass = all_mass && L.m > 0.0;
                to_lower = to_lower || std::abs(W(*A) / W(L)) > 1.0;
                A = &L;
            }
            to_lower = to_lower || std::abs(W(*A) / W(mat_far)) > 1.0;
            if (all_mass && to_lower)
            {
                const int it = is_scalar ? 4 : 1;
                const double ct_r = std::abs(std::real(ci));
                const double mc = (mass_cos >= 0.0) ? mass_cos : ct_r;
                std::complex<double> t_te(1.0, 0.0), t_tm(1.0, 0.0), a_te, a_tm;
                double x_run = mass_bonded, amp = 1.0;
                A = &mat_near;
                for (int i = 0; i <= n_layers; ++i)
                {
                    const Material &B = (i < n_layers) ? *layers[i].mtl : mat_far;
                    A->interact_with(B, it, 0.0, fGHz, &a_te, &a_tm, nullptr, nullptr, nullptr, nullptr, false, &ci, true);
                    t_te *= a_te, t_tm *= a_tm;
                    if (i < n_layers) // the mass term, formed as for a single mass-law sheet
                    {
                        double thk = layers[i].thk, dist_max = 1.0e30;
                        double g1m = B.medium_gain(1.0, fGHz, 1.0, 0.0, 0.0, 1.0e300);
                        if (g1m > 0.0)
                        {
                            double rate = -10.0 * std::log10(g1m);
                            if (rate > 0.0)
                                dist_max = 400.0 / rate;
                        }
                        else
                            dist_max = 0.0;
                        if (dist_max < thk)
                            dist_max = thk;
                        double dist = (ct_r > 0.0) ? thk / ct_r : dist_max;
                        if (!(dist < dist_max))
                            dist = dist_max;
                        double mass_path = thk * mc;
                        double gm = B.medium_gain(dist, fGHz, ct_r, mass_path, 1.0, (mass_path > 0.0) ? 0.0 : 1.0e300, thk, mc, x_run);
                        if (mass_path > 0.0)
                            x_run += B.mass_X0(thk, fGHz);
                        amp *= std::sqrt((gm < 0.0) ? 0.0 : gm);
                    }
                    A = &B;
                }
                // The far medium's interface gain is owed only where it is newly entered
                if (!mat_far.same_as(Material()) && mat_far.same_as(mat_near))
                    amp /= std::sqrt(mat_far.interface_gain(fGHz));
                std::complex<double> r_te, r_tm;
                mat_near.interact_with(*layers[0].mtl, is_scalar ? 3 : 0, 0.0, fGHz, &r_te, &r_tm, nullptr, nullptr, nullptr, nullptr,
                                       false, &ci, true);
                if (T_te)
                    *T_te = amp * t_te;
                if (T_tm)
                    *T_tm = amp * t_tm;
                if (R_te) // a mass-law layer displaces nothing, so the port has no reference plane to move
                    *R_te = r_te;
                if (R_tm)
                    *R_tm = r_tm;
                return true;
            }
        }

        const double c0 = 299792458.0;
        const double omega = 2.0 * 3.14159265358979323846 * fGHz * 1e9;
        const double k0 = omega / c0;

        auto root = [](std::complex<double> a)
        {
            if (qd_evanescent(a))
                a = std::complex<double>(std::real(a), -0.0); // decaying evanescent root
            return std::sqrt(a);
        };

        // Interface permittivities carry the resonance, the propagation index does not. That split
        // is the one the rest of this file applies at five separate sites: a permittivity resonance
        // is an interface effect rather than a propagation one.
        std::complex<double> eta_n = mat_near.eta(fGHz) + mat_near.eta_resonance(fGHz);
        std::complex<double> mu_n = mat_near.mu(fGHz);
        std::complex<double> eta_f = mat_far.eta(fGHz) + mat_far.eta_resonance(fGHz);
        std::complex<double> mu_f = mat_far.mu(fGHz);

        const std::complex<double> w_near = eta_n * mu_n; // anchors the transverse invariant
        const std::complex<double> s2 = 1.0 - ci * ci;
        const std::complex<double> z_n = std::sqrt(eta_n / mu_n);
        const std::complex<double> z_f = std::sqrt(eta_f / mu_f);
        const std::complex<double> c_far = root(1.0 - (w_near / (eta_f * mu_f)) * s2);

        // One-way loss cap, in nepers of amplitude. cos and sin of a complex phase grow like
        // exp(|Im|), where the Airy form's phi decayed instead, so an opaque layer would overflow
        // the product rather than vanish from it. 46.05 Np is 400 dB, the same opacity floor produce
        // mode caps its path length at, so nothing a propagating wave can reach is touched.
        const double dl_im_cap = 46.05;

        std::complex<double> r_pol[2], t_pol[2];
        for (int pol = 0; pol < 2 && !scattering; ++pol)
        {
            std::complex<double> M11(1.0, 0.0), M12(0.0, 0.0), M21(0.0, 0.0), M22(1.0, 0.0);
            for (int i = 0; i < n_layers; ++i)
            {
                const Material &L = *layers[i].mtl;
                std::complex<double> eta_L = L.eta(fGHz) + L.eta_resonance(fGHz);
                std::complex<double> mu_L = L.mu(fGHz);
                std::complex<double> c_L = root(1.0 - (w_near / (eta_L * mu_L)) * s2);
                std::complex<double> n_L = std::sqrt(L.eta(fGHz) * mu_L); // resonance-excluded
                std::complex<double> y_L = (pol == 0) ? (std::sqrt(eta_L / mu_L) * c_L)
                                                      : (std::sqrt(eta_L / mu_L) / c_L);

                std::complex<double> dl = k0 * n_L * c_L * layers[i].thk;

                // The alpha column, as nepers of amplitude over the path thk / |Re c|. At a
                // vanishing cosine the path is unbounded and the layer is opaque, which the cap
                // below expresses.
                double a_dB = L.alpha * std::pow(fGHz / L.fRef, L.alphaB); // dB per meter
                if (a_dB > 0.0)
                {
                    double ct_r = std::abs(std::real(c_L));
                    double a_Np = (ct_r > 0.0) ? 0.05 * std::log(10.0) * a_dB * layers[i].thk / ct_r
                                               : dl_im_cap;
                    dl -= std::complex<double>(0.0, a_Np);
                }
                if (std::imag(dl) < -dl_im_cap)
                    dl = std::complex<double>(std::real(dl), -dl_im_cap);
                else if (std::imag(dl) > dl_im_cap)
                    dl = std::complex<double>(std::real(dl), dl_im_cap);

                std::complex<double> A = std::cos(dl);
                std::complex<double> B = std::complex<double>(0.0, 1.0) * std::sin(dl) / y_L;
                std::complex<double> C = std::complex<double>(0.0, 1.0) * y_L * std::sin(dl);
                std::complex<double> D = std::cos(dl);

                std::complex<double> m11 = M11 * A + M12 * C, m12 = M11 * B + M12 * D;
                std::complex<double> m21 = M21 * A + M22 * C, m22 = M21 * B + M22 * D;
                M11 = m11, M12 = m12, M21 = m21, M22 = m22;
            }

            std::complex<double> y0 = (pol == 0) ? (z_n * ci) : (z_n / ci);
            std::complex<double> ys = (pol == 0) ? (z_f * c_far) : (z_f / c_far);
            std::complex<double> Bv = M11 + M12 * ys, Cv = M21 + M22 * ys;
            std::complex<double> den = y0 * Bv + Cv;
            if (!(std::abs(den) > 0.0))
                return false;
            r_pol[pol] = (y0 * Bv - Cv) / den;
            t_pol[pol] = 2.0 * y0 / den;

            if (is_scalar) // one channel, the TE / admittance form, as everywhere else in this file
            {
                r_pol[1] = r_pol[0], t_pol[1] = t_pol[0];
                break;
            }
        }

        std::complex<double> Rv_te = r_pol[0];
        std::complex<double> Rv_tm = is_scalar ? r_pol[0] : -r_pol[1];
        std::complex<double> tv_te = t_pol[0];
        std::complex<double> tv_tm = is_scalar ? t_pol[0] : (ci / c_far) * t_pol[1];

        if (scattering)
        {
            // Medium j: 0 is the near half-space, 1 .. n_layers the layers, n_layers + 1 the far one
            auto MED = [&](int j) -> const Material &
            { return (j == 0) ? mat_near : ((j > n_layers) ? mat_far : *layers[j - 1].mtl); };
            auto COS = [&](int j) -> std::complex<double>
            {
                if (j == 0)
                    return ci;
                const Material &X = MED(j);
                return root(1.0 - (w_near / ((X.eta(fGHz) + X.eta_resonance(fGHz)) * X.mu(fGHz))) * s2);
            };
            auto ADM = [&](int j) -> std::complex<double>
            {
                const Material &X = MED(j);
                return std::sqrt((X.eta(fGHz) + X.eta_resonance(fGHz)) / X.mu(fGHz));
            };

            // A mass-law layer in which the wave is evanescent - lossless, and the transverse
            // invariant at or above its eps mu, which is the verdict Material::interact_with
            // gives a wave that meets it - is closed on both faces. A rigid layer of the acoustic
            // mapping is that at every practical angle. The Fresnel reflectance of such a face
            // does not say so: next to a lossy medium it is below one although the wave in the
            // layer carries no flux, the remainder being the interference of the incident and
            // the reflected wave.
            auto CLOSED = [&](int j) -> bool
            {
                if (j < 1 || j > n_layers || !(MED(j).m > 0.0))
                    return false;
                const Material &X = MED(j);
                std::complex<double> w = (X.eta(fGHz) + X.eta_resonance(fGHz)) * X.mu(fGHz);
                return std::imag(w) == 0.0 && std::abs((w_near / w) * s2) >= 1.0;
            };

            std::complex<double> R_out[2], T_out[2];
            for (int pol = 0; pol < 2; ++pol)
            {
                // Normal flux of the media a leak entered a closed layer from, and left it into
                double fl_in = 1.0, fl_out = 1.0;

                // Boundary j | j+1 in produce mode's amplitudes, tf folded in by the pair that
                // forms it. A boundary neither of whose media carries tf is left exactly raw,
                // unless it is the face of a closed layer.
                //
                // A closed face passes the flat leak a bare face passes, 1 - R' with the
                // reflectance taken as one, and nothing where neither medium carries tf. The
                // reflection keeps its Fresnel magnitude. The leak is a statement about normal
                // flux, of the side that carries one, and the segments run in raw amplitudes that
                // are converted once at the outer pair; across a closed layer there is nothing
                // to convert through. So the flux of that side is collected here, per channel as
                // Material::flux_amp forms it, and the transmit port is put right by the ratio
                // below: each segment is then on flux between its own ends. Where the media on
                // the two sides of the closed layers are one, the ratio is exactly one.
                auto BND = [&](int j, std::complex<double> &r, std::complex<double> &t, std::complex<double> &tb)
                {
                    const Material &A = MED(j), &B = MED(j + 1);
                    std::complex<double> za = ADM(j), zb = ADM(j + 1), ca = COS(j), cb = COS(j + 1);
                    if (pol == 0)
                    {
                        std::complex<double> a = za * ca, b = zb * cb;
                        r = (a - b) / (a + b), t = (2.0 * a) / (a + b), tb = (2.0 * b) / (b + a);
                    }
                    else
                    {
                        r = (zb * ca - za * cb) / (zb * ca + za * cb);
                        t = (2.0 * za * ca) / (zb * ca + za * cb);
                        tb = (2.0 * zb * cb) / (za * cb + zb * ca);
                    }
                    const bool cA = CLOSED(j), cB = CLOSED(j + 1);
                    const bool closed = cA != cB; // a closed layer against a side that carries flux
                    if (A.tf(fGHz) == 0.0 && B.tf(fGHz) == 0.0 && !closed)
                        return;
                    double R0 = std::norm(r);
                    double Rt = A.apply_tf_pair(B, R0, fGHz);
                    double keep = 1.0 - R0, pass = (Rt < 1.0) ? 1.0 - Rt : 0.0;
                    r = std::polar(std::sqrt(Rt), std::arg(r));
                    if (keep < 1.0e-9) // no forward Fresnel port left to scale: the flat leak
                        t = std::polar(std::sqrt(pass), std::arg(t)), tb = std::polar(std::sqrt(pass), std::arg(tb));
                    else if (closed) // no forward port either, whatever R0 reads: the leak of a bare face
                    {
                        double Rc = A.apply_tf_pair(B, 1.0, fGHz);
                        pass = (Rc < 1.0) ? 1.0 - Rc : 0.0;
                        t = std::polar(std::sqrt(pass), std::arg(t)), tb = std::polar(std::sqrt(pass), std::arg(tb));
                    }
                    else
                        t *= std::sqrt(pass / keep), tb *= std::sqrt(pass / keep);
                    if (closed)
                    {
                        int o = cA ? j + 1 : j; // the side that carries flux
                        double g = (pol == 0) ? std::real(ADM(o) * COS(o)) : std::real(std::conj(ADM(o)) * COS(o));
                        (cB ? fl_in : fl_out) *= g;
                    }
                };

                // One-way factor of coherent layer j, the same phase the matrix is built from
                auto PHI = [&](int j) -> std::complex<double>
                {
                    const Material &L = MED(j);
                    std::complex<double> c_L = COS(j);
                    std::complex<double> dl = k0 * std::sqrt(L.eta(fGHz) * L.mu(fGHz)) * c_L * layers[j - 1].thk;
                    double a_dB = L.alpha * std::pow(fGHz / L.fRef, L.alphaB);
                    if (a_dB > 0.0)
                    {
                        double ct_r = std::abs(std::real(c_L));
                        dl -= std::complex<double>(0.0, (ct_r > 0.0) ? 0.05 * std::log(10.0) * a_dB * layers[j - 1].thk / ct_r
                                                                     : dl_im_cap);
                    }
                    if (std::imag(dl) < -dl_im_cap)
                        dl = std::complex<double>(std::real(dl), -dl_im_cap);
                    else if (std::imag(dl) > dl_im_cap)
                        dl = std::complex<double>(std::real(dl), dl_im_cap);
                    return std::exp(std::complex<double>(0.0, -1.0) * dl);
                };

                // Real amplitude factor of mass-law layer j, formed as produce mode forms it: the
                // loss over the path thk / |Re c| under the same saturation cap, and the mass term
                // from the normal thickness and the cosine the wave has in air, which the caller
                // hands in as mass_cos where the near medium is not air.
                auto MASS = [&](int j, double &x_run) -> double
                {
                    const Material &L = MED(j);
                    double thk = layers[j - 1].thk, ct_r = std::abs(std::real(COS(j)));
                    double dist_max = 1.0e30;
                    double g1m = L.medium_gain(1.0, fGHz, 1.0, 0.0, 0.0, 1.0e300);
                    if (g1m > 0.0)
                    {
                        double rate = -10.0 * std::log10(g1m);
                        if (rate > 0.0)
                            dist_max = 400.0 / rate;
                    }
                    else
                        dist_max = 0.0;
                    if (dist_max < thk)
                        dist_max = thk;
                    double dist = (ct_r > 0.0) ? thk / ct_r : dist_max;
                    if (!(dist < dist_max))
                        dist = dist_max;
                    double mc = (mass_cos >= 0.0) ? mass_cos : std::abs(std::real(ci));
                    double mass_path = thk * mc;
                    double g = L.medium_gain(dist, fGHz, ct_r, mass_path, 1.0, (mass_path > 0.0) ? 0.0 : 1.0e300, thk, mc, x_run);
                    if (mass_path > 0.0)
                        x_run += L.mass_X0(thk, fGHz);
                    return std::sqrt((g < 0.0) ? 0.0 : g);
                };

                std::complex<double> Tt(1.0, 0.0), Rt(0.0, 0.0);
                int p = 0;
                bool first = true;
                double x_run = mass_bonded; // X0 of the plate crossed so far
                for (int q = 1; q <= n_layers + 1; ++q)
                {
                    bool mass = (q <= n_layers) && MED(q).m > 0.0;
                    if (q <= n_layers && !mass)
                        continue;
                    if (q - p > 1) // a coherent layer separates what follows from the plate
                        x_run = 0.0;

                    // Coherent segment between media p and q, from its far boundary inwards. G is
                    // the reflection the series runs on. Ge is the one the reflect port hands
                    // out: what entered a layer and came back paid that layer's interface gain,
                    // once, as the transmit port pays it once; the series itself pays nothing.
                    std::complex<double> G, Ge, T, r, t, tb;
                    BND(q - 1, G, T, tb);
                    Ge = G;
                    for (int j = q - 2; j >= p; --j)
                    {
                        std::complex<double> phi = PHI(j + 1), p2 = phi * phi;
                        BND(j, r, t, tb);
                        std::complex<double> den = 1.0 + r * G * p2;
                        if (!(std::abs(den) > 0.0))
                            return false;
                        T = t * phi * T / den;
                        Ge = r + std::sqrt(MED(j + 1).interface_gain(fGHz)) * t * tb * Ge * p2 / den;
                        G = r + t * tb * G * p2 / den;
                    }
                    Tt *= T;
                    if (first)
                        Rt = Ge, first = false;
                    if (mass)
                        Tt *= MASS(q, x_run);
                    p = q;
                }
                if (fl_in != fl_out) // one medium on both sides of the closed layers: exactly one
                    Tt *= (fl_in > 0.0 && fl_out > 0.0) ? std::sqrt(fl_in / fl_out) : 0.0;
                R_out[pol] = Rt, T_out[pol] = Tt;
                if (is_scalar)
                {
                    R_out[1] = R_out[0], T_out[1] = T_out[0];
                    break;
                }
            }
            Rv_te = R_out[0], Rv_tm = R_out[1], tv_te = T_out[0], tv_tm = T_out[1];
        }

        // The one conversion to per-channel normal flux, at the outer pair, transmit port only. The
        // gate is near == far, so an isolated stack, a stack embedded in one medium and a collapsed
        // duplicate all take it as exactly 1, matching produce mode.
        double fx_te = 1.0, fx_tm = 1.0;
        if (!mat_far.same_as(mat_near))
        {
            Material::flux_amp(z_n, z_f, ci, c_far, fx_te, fx_tm);
            if (is_scalar)
                fx_tm = fx_te;
        }

        // Lumped interface gains: one per layer entered, plus the far medium's when the wave enters
        // a medium it was not already in. The generalization of produce mode's ig to a stack.
        bool far_is_air = mat_far.same_as(Material());
        bool far_is_new = !far_is_air && !mat_far.same_as(mat_near);
        double ig2 = far_is_new ? mat_far.interface_gain(fGHz) : 1.0;
        for (int i = 0; i < n_layers; ++i)
            ig2 *= layers[i].mtl->interface_gain(fGHz);
        double ig = std::sqrt((ig2 < 0.0) ? 0.0 : ig2);

        // The displaced host, over the stack's TOTAL normal thickness. A stack of sheets has no
        // world-space thickness either, so the tracer walked the whole of it as the host medium and
        // the transmit port owes the same division a single sheet owes. disp_budget caps it exactly
        // as it caps a single sheet, which matters more here: a two-layer stack reaches a budget a
        // one-layer one would not.
        //
        // A mass-law layer displaces nothing, as a single mass-law sheet displaces nothing: it has
        // no in-layer phase to set against the host's.
        double t_tot = 0.0;
        for (int i = 0; i < n_layers; ++i)
            if (!(layers[i].mtl->m > 0.0))
                t_tot += layers[i].thk;
        double t_disp = (disp_budget >= 0.0 && disp_budget < t_tot) ? disp_budget : t_tot;
        if (t_disp < 0.0)
            t_disp = 0.0;

        const Material &disp_mtl = disp_far ? mat_far : mat_near;
        std::complex<double> eta_d = disp_mtl.eta(fGHz) + disp_mtl.eta_resonance(fGHz);
        std::complex<double> mu_d = disp_mtl.mu(fGHz);
        std::complex<double> n_d = std::sqrt(disp_mtl.eta(fGHz) * mu_d); // resonance-excluded
        std::complex<double> c_disp = root(1.0 - (w_near / (eta_d * mu_d)) * s2);
        std::complex<double> phi_disp =
            std::exp(std::complex<double>(0.0, -1.0) * k0 * n_d * t_disp * c_disp);
        std::complex<double> undisp = (std::abs(phi_disp) > 0.0)
                                          ? std::complex<double>(1.0, 0.0) / phi_disp
                                          : std::complex<double>(1.0, 0.0);

        // The reflect port is handed to a ray that turns around at the mesh plane, so it owes the
        // round trip through the part of the stack that lies in front of that plane: half of it
        // where one medium is on both sides, the mesh plane being the mid-plane; all of it where
        // the stack lies in the near medium in front of a solid; none where it lies beyond.
        std::complex<double> front = mat_far.same_as(mat_near) ? undisp
                                                               : (disp_far ? std::complex<double>(1.0, 0.0) : undisp * undisp);

        if (T_te)
            *T_te = ig * fx_te * tv_te * undisp;
        if (T_tm)
            *T_tm = ig * fx_tm * tv_tm * undisp;
        if (R_te)
            *R_te = front * Rv_te;
        if (R_tm)
            *R_tm = front * Rv_tm;
        return true;
    }

    // Analytic thin-slab (Fabry-Perot) resolve for a near | slab | far stack. Shared by both public
    // functions in this file: ray_state_update corrects coefficients it has already written, while
    // ray_mesh_interact produces them outright, because a sheet has zero world-space thickness and
    // there is nowhere for the tracer to re-emit to.
    //
    // Direction convention: mat_near is the medium the incident wave arrives from, mat_far the medium
    // the transmitted wave leaves into. Correction mode is symmetric in the two - near and far appear
    // only as rn*rf, Rn*Rf and (1-Rn)*(1-Rf) - which is why ray_state_update may pass them in its own
    // "interface ahead of the ray / interface it entered through" order. Produce mode is NOT
    // symmetric: the reflect port and the interface gains both depend on which side the wave is on.
    //
    // abs_cos_theta_t is the in-slab propagation cosine and abs_cos_theta the incidence cosine at
    // the outer face, in both modes. The two length arguments differ by mode:
    //
    //   produce == false: dist_refract and dist_geo are the traced path lengths through the layer,
    //                     refracted and geometric, exactly as the tracer measured them.
    //   produce == true : dist_refract is the sheet's NORMAL thickness and dist_geo is unused. A
    //                     sheet has no traced path, so the caller has nothing to measure and the one
    //                     quotient the traversal needs, thk / cos, is formed here instead. It used to
    //                     be formed by the caller under a geometric floor on the cosine; see the
    //                     saturation note at the top of the body for why that floor is gone.
    //
    // produce == false: T_te / T_tm are the per-channel series factor S, and nothing else. The port
    //                   the caller holds already carries the per-channel Fresnel coefficient of its
    //                   own interface, so no magnitude conversion is owed. In scalar mode both outputs
    //                   carry the same factor, because the caller scales a stored xprmat in place
    //                   instead of rebuilding it from cTE. Both boundaries are evaluated at the real
    //                   in-slab angle, unless the caller supplies ct_slab_exact - the exact complex
    //                   value recovered from the traced direction - in which case both boundaries and
    //                   the propagation factor use that instead.
    // produce == true : T_te / T_tm and R_te / R_tm are absolute composite coefficients. The
    //                   transmit port carries the slab's interface gain, plus the far medium's when
    //                   the far medium is one the wave newly enters, matching what
    //                   Material::interact_with folds in for a transmission-class interaction. The
    //                   reflect port carries the slab's on the part that entered the slab, and the
    //                   round trip through the part of the layer in front of the mesh plane. A
    //                   caller that also passes the far medium through
    //                   separately (the layer-2 forward path of a sheet stack) must not pay that
    //                   interface gain twice.
    //                   Both boundaries are evaluated at the true complex angles, which the incidence
    //                   angle alone determines - nothing propagates geometrically inside a sheet, so
    //                   there is no real direction to project onto. The interface amplitudes are the
    //                   raw Fresnel ones, rescaled only by the tf redistribution, so that a chain of
    //                   crossings telescopes into the correct end-to-end amplitude, and the
    //                   propagation factor is exp(-j k0 n thk cos_theta_slab) with both quantities
    //                   complex - magnitude and phase out of one expression, with only the alpha
    //                   column drawn from the material on top. The transmit port also subtracts the
    //                   propagation of the medium it displaces over the same thickness, since the
    //                   tracer walked that length as that medium and the rest of the path already
    //                   carries it. Together these put the composite on the exact analytic slab at
    //                   every angle. disp_far and disp_budget control the displacement, see below.
    //
    // Returns false only in correction mode, meaning "did not resolve, let the tracer follow the
    // bounces". Produce mode always resolves.
    inline bool qd_slab_resolve(const Material &mat_near,
                                const Material &mat_slab,
                                const Material &mat_far,
                                double dist_refract, // correction: traced refracted path; produce: normal thickness
                                double dist_geo,     // correction: traced geometric path; produce: unused
                                double abs_cos_theta_t, double abs_cos_theta,
                                double fGHz, bool is_scalar, double eps, bool produce,
                                std::complex<double> *T_te, std::complex<double> *T_tm,
                                std::complex<double> *R_te = nullptr,
                                std::complex<double> *R_tm = nullptr,
                                bool disp_far = false,     // sheet sits on the far side (glued exit)
                                double disp_budget = -1.0, // max normal thickness it may replace, < 0 = all
                                std::complex<double> ct_slab_exact = std::complex<double>(0.0, 0.0),
                                std::complex<double> ct_near_exact = std::complex<double>(0.0, 0.0),
                                const Material *back_near = nullptr, // what lies behind a sheet named as mat_near
                                bool far_closed = false,             // produce: near | far is past its critical angle
                                bool near_undeviated = false,        // produce: the wave in mat_near was not refracted in
                                double mass_bonded = 0.0,            // produce: X0 of the bonded mass-law bodies already crossed
                                const Material *back_far = nullptr,  // correction: what lies behind a sheet named as mat_far
                                bool verdict_given = false,          // correction: the caller decided to sum; eps is not read
                                double mass_cos = -1.0)              // produce: angle factor of the mass term, < 0 = abs_cos_theta
    {
        // Mass-law materials carry a lumped transmission surrogate: eta/mu are calibrated for
        // surface impedance only, so the in-slab index (and hence the phase of phi) is not
        // physical and the internal series must not be resummed. Correction mode re-emits; the
        // tracer follows the bounces and each traversal pays medium_gain once. Produce mode has no
        // interior to re-emit into and takes the incoherent route below instead.
        if (!produce && mat_slab.m > 0.0)
            return false;

        const double c0 = 299792458.0;
        const double omega = 2.0 * 3.14159265358979323846 * fGHz * 1e9;

        bool far_is_air = mat_far.same_as(Material());
        std::complex<double> eta_s_if = mat_slab.eta(fGHz) + mat_slab.eta_resonance(fGHz);
        std::complex<double> mu_s = mat_slab.mu(fGHz);
        double n_cur = std::real(std::sqrt(mat_slab.eta(fGHz) * mu_s)); // resonance-excluded slab index

        // Traversal geometry, in the form the rest of the function needs.
        //
        // The physics is written in the normal thickness and a cosine: the propagation factor is
        // exp(-j k0 n thk_normal cos_slab) and the mass-law term is thk_normal * cos_theta. Neither
        // diverges at any angle. The path length thk / cos does diverge, and it is the only quantity
        // here that can, so it is formed once and only where a genuine path length is owed - the
        // alpha column and the mass-law conductivity term.
        //
        // Both of those are dB-linear in the path, so the quotient saturates: past the length at
        // which the layer is opaque nothing downstream can move. dist_max is that length, read off
        // the material's own loss per meter, and it is never shorter than the normal thickness so the
        // cap cannot reach a result at any angle a propagating wave has. It depends on the material
        // and the frequency and on no geometry at all, which is what a geometric floor on the cosine
        // could not offer.
        //
        // Correction mode is handed genuine traced lengths and keeps them.
        const double thk_normal = produce ? dist_refract : dist_refract * abs_cos_theta_t;
        double dist_slab = dist_refract;
        double dist_max = 1.0e30;
        if (produce)
        {
            double g1m = mat_slab.medium_gain(1.0, fGHz, 1.0, 0.0, 0.0, 1.0e300); // 1 m, mass term off
            if (g1m > 0.0)
            {
                double rate = -10.0 * std::log10(g1m); // dB per meter
                if (rate > 0.0)
                    dist_max = 400.0 / rate;
            }
            else
                dist_max = 0.0; // opaque within a meter
            if (dist_max < thk_normal)
                dist_max = thk_normal;

            dist_slab = (abs_cos_theta_t > 0.0) ? thk_normal / abs_cos_theta_t : dist_max;
            if (!(dist_slab < dist_max)) // also catches inf and NaN
                dist_slab = dist_max;
        }
        // Slab-side propagation cosine for correction mode. The tracer's direction is real, so
        // normally only the real projection of it is available and every boundary gets re-derived from
        // that projection - including the entry boundary, whose coefficient ray_mesh_interact had
        // already applied exactly. The Airy denominator then sums a series against a coefficient the
        // path never carried. When the caller has recovered the exact complex value by inverting the
        // refraction map it passes it here, and both boundaries and the propagation factor use it.
        bool have_exact_ct = std::abs(ct_slab_exact) > 0.0;
        std::complex<double> cs = have_exact_ct ? ct_slab_exact
                                                : std::complex<double>(abs_cos_theta_t, 0.0);
        std::complex<double> sin2 = 1.0 - cs * cs; // in-slab sine (fbs_angleN convention)

        // Incidence cosine in the near medium, produce mode only.
        //
        // abs_cos_theta is a real angle and is what a face hit supplies. A sheet reached from inside
        // a traced medium has no real incidence cosine: the propagation cosine there is complex and
        // the tracer's direction carries only its real projection. Such a caller passes the
        // recovered value through ct_near_exact, and the projection is then unused. Every expression
        // below is analytic in it, so a real value reproduces the angle path exactly; where the
        // caller supplies nothing, sin2_i keeps the real spelling verbatim so nothing moves.
        //
        // Correction mode never reads either of these. Its angle comes from real ray geometry and
        // its result must not move.
        bool have_exact_ci = std::abs(ct_near_exact) > 0.0;
        std::complex<double> ci = have_exact_ci ? ct_near_exact
                                                : std::complex<double>(abs_cos_theta, 0.0);
        std::complex<double> sin2_i =
            have_exact_ci ? (1.0 - ci * ci)
                          : std::complex<double>(1.0 - abs_cos_theta * abs_cos_theta, 0.0);

        // Exact complex in-slab cosine, derived from the incidence cosine in the near medium.
        // Produce mode holds that cosine, so nothing has to be projected onto a real direction: the
        // far boundary and the propagation factor both use this. Snell is transitive, so with the
        // recovered near-side cosine this is the same value a bare entry | slab crossing would have
        // used, and the host drops out of it. Correction mode has only the traced real angle and
        // keeps the real form throughout.
        std::complex<double> cts_exact(1.0, 0.0);
        if (produce)
        {
            std::complex<double> eta_n_if = mat_near.eta(fGHz) + mat_near.eta_resonance(fGHz);
            std::complex<double> mu_n = mat_near.mu(fGHz);
            std::complex<double> arg = 1.0 - ((eta_n_if * mu_n) / (eta_s_if * mu_s)) * sin2_i;
            if (qd_evanescent(arg))
                arg = std::complex<double>(std::real(arg), -0.0); // decaying evanescent root
            cts_exact = std::sqrt(arg);
        }

        // Fresnel TE/TM amplitude reflection at slab|adjacent from the slab side, tf folded into |r|.
        // t_out returns the raw Fresnel transmission amplitude for the slab -> adjacent crossing; only
        // its phase is used, since tf redistributes energy between the ports and the raw magnitude is
        // no longer the transmitted one.
        auto fresnel_r = [&](const Material &adj,
                             std::complex<double> &r_te, std::complex<double> &r_tm,
                             double &Ra_te, double &Ra_tm,
                             std::complex<double> *t_out = nullptr, // [te, tm] slab -> adjacent
                             const Material *backing = nullptr)     // what lies behind adj if it is a sheet
        {
            std::complex<double> eta_a_if = adj.eta(fGHz) + adj.eta_resonance(fGHz);
            std::complex<double> mu_a = adj.mu(fGHz);
            std::complex<double> z1 = std::sqrt(eta_s_if / mu_s);
            std::complex<double> z2 = std::sqrt(eta_a_if / mu_a);
            std::complex<double> rat = (eta_s_if * mu_s) / (eta_a_if * mu_a);

            // Past the internal critical angle the argument is negative real and the principal root
            // is the growing evanescent wave, which flips the sign of the internal reflection phase
            // under the exp(-i k d) convention phi uses below. Force the decaying root, exactly as
            // interact_with does. Magnitudes are unaffected (|r| = 1 either way), so this only moves
            // results that actually reach internal TIR.
            std::complex<double> ct2_arg = 1.0 - rat * sin2;
            if (qd_evanescent(ct2_arg))
                ct2_arg = std::complex<double>(std::real(ct2_arg), -0.0);
            std::complex<double> cos_t2 = std::sqrt(ct2_arg);

            std::complex<double> te = (z1 * cs - z2 * cos_t2) / (z1 * cs + z2 * cos_t2);
            std::complex<double> tm = is_scalar ? te : (z2 * cs - z1 * cos_t2) / (z2 * cs + z1 * cos_t2);

            // A sheet named as a bounding medium is a layer of finite thickness, not a half-space.
            // Reading only its eta and mu says the wave enters it and never comes back, which is
            // exact only if whatever sits behind the sheet has the sheet's own permittivity. For a
            // sheet electrically close to the slab it is worse than inexact: the half-space
            // reflection goes to zero, the Airy denominator collapses to 1, and the slab loses its
            // internal series, when physically the wave passes through the sheet almost unimpeded
            // and still reflects off the boundary behind it.
            //
            // The backing is what the state format cannot always name. Where the caller knows it,
            // it passes it; where it does not, air is assumed. Air is the right default for the
            // motivating geometry, a coating on the room side of a wall, and it is what the
            // characteristic-matrix reference in the fixture encodes. Naming it from the state
            // rather than assuming it is its own open item.
            //
            // The layer reflection already carries tf: the core applies the pair rule at the
            // slab | sheet face and at the sheet | backing face, so nothing is redistributed a
            // second time here and only the [0, 1] guard remains.
            //
            // A mass-law sheet is the exception and keeps the half-space value. That is not a
            // fallback: its reflect port is its near interface and nothing else, by the mass-law
            // convention - no series, nothing returning from behind it - which is exactly what
            // produce mode returns for the same sheet.
            bool layer_resolved = false;
            if (adj.thk > 0.0 && !(adj.m > 0.0))
            {
                QdLayer lay;
                lay.mtl = &adj, lay.thk = adj.thk;
                Material air_backing;
                std::complex<double> Lr_te, Lr_tm;
                // Seen from the slab a lining lies in the air beyond it; a sheet inside the slab
                // has the slab on both sides and the core centres it.
                if (qd_stack_resolve(mat_slab, &lay, 1, backing ? *backing : air_backing,
                                     cs, fGHz, is_scalar, nullptr, nullptr, &Lr_te, &Lr_tm, true))
                    te = Lr_te, tm = is_scalar ? Lr_te : Lr_tm, layer_resolved = true;
            }

            // tf of the boundary: both media that form it, symmetrically. With no tf on the
            // neighbour this is the slab's own apply_tf, bit for bit.
            if (layer_resolved)
            {
                Material no_tf;
                Ra_te = no_tf.apply_tf(std::norm(te), fGHz);
                Ra_tm = no_tf.apply_tf(std::norm(tm), fGHz);
            }
            else
            {
                Ra_te = mat_slab.apply_tf_pair(adj, std::norm(te), fGHz);
                Ra_tm = mat_slab.apply_tf_pair(adj, std::norm(tm), fGHz);
            }
            r_te = std::polar(std::sqrt(Ra_te), std::arg(te));
            r_tm = std::polar(std::sqrt(Ra_tm), std::arg(tm));

            if (t_out) // ITU-R P.2040-1 eq. (32) with the slab as the incidence medium
                t_out[0] = (2.0 * z1 * cs) / (z1 * cs + z2 * cos_t2),
                t_out[1] = is_scalar ? t_out[0] : (2.0 * z1 * cs) / (z2 * cs + z1 * cos_t2);
        };

        // The near interface seen from the incidence side, at the true incidence angle rather than at
        // the real-projected in-slab angle. The two views are reciprocal only for a lossless slab: in a
        // lossy one the in-slab refraction cosine is complex, and dropping its imaginary part costs
        // exact reciprocity - by 3 % at 80 deg for sigma = 0.02 S/m, far more for a conductive slab.
        // A traced solid evaluates its entry interface from the incidence side, so a sheet has to do
        // the same to stay equivalent to one. Correction mode keeps the slab-side view, because
        // ray_state_update's angle comes from real ray geometry and its result must not move.
        auto fresnel_r_near = [&](std::complex<double> &r_te, std::complex<double> &r_tm,
                                  double &Ra_te, double &Ra_tm,
                                  double &R0_te, double &R0_tm, // raw, pre-tf, for the amplitude scale
                                  std::complex<double> *t_fwd,  // [te, tm] near -> slab
                                  std::complex<double> *t_bwd)  // [te, tm] slab -> near
        {
            std::complex<double> eta_a_if = mat_near.eta(fGHz) + mat_near.eta_resonance(fGHz);
            std::complex<double> mu_a = mat_near.mu(fGHz);
            std::complex<double> z_a = std::sqrt(eta_a_if / mu_a);
            std::complex<double> z_s = std::sqrt(eta_s_if / mu_s);
            std::complex<double> rat = (eta_a_if * mu_a) / (eta_s_if * mu_s);

            std::complex<double> ct2_arg = 1.0 - rat * sin2_i;
            if (qd_evanescent(ct2_arg))
                ct2_arg = std::complex<double>(std::real(ct2_arg), -0.0); // decaying evanescent root
            std::complex<double> cos_t2 = std::sqrt(ct2_arg);

            std::complex<double> te = (z_a * ci - z_s * cos_t2) / (z_a * ci + z_s * cos_t2);
            std::complex<double> tm = is_scalar ? te : (z_s * ci - z_a * cos_t2) / (z_s * ci + z_a * cos_t2);
            R0_te = std::norm(te), R0_tm = std::norm(tm);
            Ra_te = mat_slab.apply_tf_pair(mat_near, R0_te, fGHz); // the pair that forms the face
            Ra_tm = mat_slab.apply_tf_pair(mat_near, R0_tm, fGHz);

            // Returned as the slab-side reflection, so that rn keeps one meaning throughout: the
            // internal series and the front port then differ only by the Stokes sign.
            r_te = std::polar(std::sqrt(Ra_te), std::arg(-te));
            r_tm = std::polar(std::sqrt(Ra_tm), std::arg(-tm));

            t_fwd[0] = (2.0 * z_a * ci) / (z_a * ci + z_s * cos_t2);
            t_fwd[1] = is_scalar ? t_fwd[0] : (2.0 * z_a * ci) / (z_s * ci + z_a * cos_t2);
            t_bwd[0] = (2.0 * z_s * cos_t2) / (z_s * cos_t2 + z_a * ci);
            t_bwd[1] = is_scalar ? t_bwd[0] : (2.0 * z_s * cos_t2) / (z_a * cos_t2 + z_s * ci);
        };

        // The far interface at the true complex in-slab cosine rather than at its real projection.
        // Produce mode holds the real incidence angle in the near medium, so the in-slab cosine follows
        // from Snell with nothing projected away, and Snell is transitive: the cosine in the far medium
        // depends only on the near and far media, not on the slab between them. Dropping the imaginary
        // part here was worth 2 dB at 85 deg for a 10 cm concrete slab and is the whole of the sheet's
        // remaining error at grazing incidence. Correction mode keeps the real-angle path above,
        // because ray_state_update's angle comes from real ray geometry and its result must not move.
        auto fresnel_r_far = [&](std::complex<double> &r_te, std::complex<double> &r_tm,
                                 double &Ra_te, double &Ra_tm,
                                 double &R0_te, double &R0_tm, // raw, pre-tf, for the amplitude scale
                                 std::complex<double> *t_out)  // [te, tm] slab -> far
        {
            std::complex<double> eta_n_if = mat_near.eta(fGHz) + mat_near.eta_resonance(fGHz);
            std::complex<double> mu_n = mat_near.mu(fGHz);
            std::complex<double> eta_a_if = mat_far.eta(fGHz) + mat_far.eta_resonance(fGHz);
            std::complex<double> mu_a = mat_far.mu(fGHz);
            std::complex<double> z_s = std::sqrt(eta_s_if / mu_s);
            std::complex<double> z_a = std::sqrt(eta_a_if / mu_a);

            // In-slab cosine from the hoisted value; the cosine in the far medium follows from the
            // same incidence-side sine, since Snell is transitive and the slab drops out of it.
            std::complex<double> cts = cts_exact;
            std::complex<double> cos_t2_arg = 1.0 - ((eta_n_if * mu_n) / (eta_a_if * mu_a)) * sin2_i;
            if (qd_evanescent(cos_t2_arg))
                cos_t2_arg = std::complex<double>(std::real(cos_t2_arg), -0.0); // decaying evanescent root
            std::complex<double> cos_t2 = std::sqrt(cos_t2_arg);

            std::complex<double> te = (z_s * cts - z_a * cos_t2) / (z_s * cts + z_a * cos_t2);
            std::complex<double> tm = is_scalar ? te : (z_a * cts - z_s * cos_t2) / (z_a * cts + z_s * cos_t2);
            R0_te = std::norm(te), R0_tm = std::norm(tm);
            Ra_te = mat_slab.apply_tf_pair(mat_far, R0_te, fGHz); // the pair that forms the face
            Ra_tm = mat_slab.apply_tf_pair(mat_far, R0_tm, fGHz);
            r_te = std::polar(std::sqrt(Ra_te), std::arg(te));
            r_tm = std::polar(std::sqrt(Ra_tm), std::arg(tm));

            t_out[0] = (2.0 * z_s * cts) / (z_s * cts + z_a * cos_t2);
            t_out[1] = is_scalar ? t_out[0] : (2.0 * z_s * cts) / (z_a * cts + z_s * cos_t2);
        };

        std::complex<double> rn_te, rn_tm, rf_te, rf_tm;
        std::complex<double> tn_fwd[2], tn_bwd[2], tf_out[2];
        double Rn_te = 0.0, Rn_tm = 0.0, Rf_te = 0.0, Rf_tm = 0.0;
        double R0n_te = 0.0, R0n_tm = 0.0, R0f_te = 0.0, R0f_tm = 0.0;
        if (produce)
        {
            fresnel_r_near(rn_te, rn_tm, Rn_te, Rn_tm, R0n_te, R0n_tm, tn_fwd, tn_bwd);
            fresnel_r_far(rf_te, rf_tm, Rf_te, Rf_tm, R0f_te, R0f_tm, tf_out);
        }
        else
        {
            fresnel_r(mat_near, rn_te, rn_tm, Rn_te, Rn_tm, nullptr, back_near);
            fresnel_r(mat_far, rf_te, rf_tm, Rf_te, Rf_tm, nullptr, back_far);
        }

        // Interface amplitudes for the produce ports: the raw Fresnel amplitude, rescaled only by
        // whatever tf redistributes between the reflected and transmitted ports.
        //
        // The magnitude used to be sqrt(1 - R). That is not the normal-flux transmittance, which is
        // Re(z2 c2)/Re(z1 c1) |t|^2 per channel; the two coincide only when the incidence medium is
        // lossless, and neither of the two internal crossings here has a lossless incidence side.
        // It is also not the single-pass amplitude of a two-interface stack: |t12 * t21| differs
        // from (1 - |r|^2) by the phase between z1*c1 and z2*c2. Measured against an exact transfer
        // matrix, that one substitution was the entire residual of the sheet composite - 0.0156 dB
        // at normal incidence for lossy concrete, against 2.5e-5 dB from every other approximation
        // here combined.
        //
        // The raw amplitudes stay, because the intermediate slab flux cancels between the two
        // crossings and never appears in the composite. What the composite does owe is the obliquity
        // factor of the pair it actually spans, near to far, which is applied once below.
        std::complex<double> tns_te, tns_tm, tsn_te, tsn_tm, tsf_te, tsf_tm;
        double ig = 1.0, ig_echo = 1.0;  // ig_echo: the sheet's own gain, on what entered it and came back
        double fx_te = 1.0, fx_tm = 1.0; // near -> far normal-flux conversion, transmit port only
        if (produce)
        {
            auto AMP = [](double R_tf, double R0, const std::complex<double> &t) -> std::complex<double>
            {
                double keep = 1.0 - R0;
                if (keep < 1.0e-9) // TIR or grazing: no forward port left to scale
                    return std::polar(std::sqrt((R_tf < 1.0) ? 1.0 - R_tf : 0.0), std::arg(t));
                return t * std::sqrt(((R_tf < 1.0) ? 1.0 - R_tf : 0.0) / keep);
            };
            tns_te = AMP(Rn_te, R0n_te, tn_fwd[0]), tns_tm = AMP(Rn_tm, R0n_tm, tn_fwd[1]); // near -> slab
            tsn_te = AMP(Rn_te, R0n_te, tn_bwd[0]), tsn_tm = AMP(Rn_tm, R0n_tm, tn_bwd[1]); // slab -> near
            tsf_te = AMP(Rf_te, R0f_te, tf_out[0]), tsf_tm = AMP(Rf_tm, R0f_tm, tf_out[1]); // slab -> far

            // The far medium's lumped interface_gain is owed only when the wave actually enters a new
            // medium, which is what Material::interact_with folds in for a transmission-class
            // interaction. Testing for air was a proxy for that, and the proxy fails for a sheet
            // embedded in a medium: with near == far the ray was already inside the far medium and
            // must not pay its entry loss a second time. Air-bounded and air|sheet|solid stacks are
            // unaffected, and solid|sheet|air gives air's interface_gain of 1.0 on either test.
            bool far_is_new = !far_is_air && !mat_far.same_as(mat_near);
            ig = std::sqrt(mat_slab.interface_gain(fGHz) * (far_is_new ? mat_far.interface_gain(fGHz) : 1.0));
            ig_echo = std::sqrt(mat_slab.interface_gain(fGHz));

            // Normal-flux conversion for the transmit port, matching what Material::interact_with
            // applies to a single crossing. The composite carries the wave from the near medium into
            // the far one, so it owes that pair's obliquity factor exactly once; the flux inside the
            // slab cancels between the two internal crossings and never enters. The reflect port
            // turns around in the near medium, where the factor is 1, and is left alone.
            //
            // With near == far the factor is identically 1, so an isolated sheet, a collapsed
            // duplicate and a sheet embedded in one medium are all untouched. Only the glued cases
            // move. Snell is transitive, so the cosine in the far medium follows from the incidence
            // sine and the slab drops out of it.
            if (!mat_far.same_as(mat_near))
            {
                std::complex<double> eta_n_if = mat_near.eta(fGHz) + mat_near.eta_resonance(fGHz);
                std::complex<double> mu_n = mat_near.mu(fGHz);
                std::complex<double> eta_f_if = mat_far.eta(fGHz) + mat_far.eta_resonance(fGHz);
                std::complex<double> mu_f = mat_far.mu(fGHz);
                std::complex<double> z_n = std::sqrt(eta_n_if / mu_n);
                std::complex<double> z_f = std::sqrt(eta_f_if / mu_f);

                std::complex<double> cf_arg = 1.0 - ((eta_n_if * mu_n) / (eta_f_if * mu_f)) * sin2_i;
                if (qd_evanescent(cf_arg))
                    cf_arg = std::complex<double>(std::real(cf_arg), -0.0); // decaying evanescent root
                std::complex<double> c_far = std::sqrt(cf_arg);

                // The denominator is the incidence-side normal flux, Re(z_n c_n) per channel. It has
                // to be the same real number the crossing that entered the near medium put in its
                // numerator, or the pair does not telescope; that is what makes the recovered cosine
                // matter here and not only in the amplitudes.
                Material::flux_amp(z_n, z_f, ci, c_far, fx_te, fx_tm);
                if (is_scalar)
                    fx_tm = fx_te;
            }
        }

        // Transmit port where the outer pair is past its critical angle.
        //
        // Total reflection of near | far says that no Fresnel wave propagates in the far medium,
        // and nothing about the sheet. The reflect port is untouched by it: the wave enters the
        // sheet, turns around at the far face and comes back, which the expressions below already
        // say. The transmit port has no Fresnel part left, since the near-to-far flux conversion
        // is zero. What crosses is the tf leak, and a leak that crosses the sheet pays what the
        // sheet charges. It is written as the three events a traced solid of the same material
        // would produce:
        //
        //     near | sheet   Material::interact_with's transmission, with the sheet's interface gain
        //     the sheet      its one-way factor below, or the mass term on the mass-law route
        //     sheet | far    Material::interact_with's transmission, with the far medium's gain
        //
        // Each event is evaluated at the cosine the wave has there, and that depends on how it
        // entered the sheet.
        //
        // Entered as a leak, near | sheet itself past its critical angle: the wave kept the
        // incidence cosine, and meets the far face with it as a wave held undeviated.
        //
        // Entered by an ordinary crossing: the wave was refracted and is at the Snell cosine. Snell
        // is transitive, so the far face is then closed whenever the outer pair is, and it passes
        // the flat leak of that face or, with no tf on it, nothing. Judging it at the incidence
        // cosine instead finds it open wherever the sheet is rarer than the near medium and writes
        // a Fresnel crossing into a medium that carries no propagating wave. The verdict is forced
        // for that reason rather than re-derived from a complex cosine, and in scalar mode the
        // phase of the leak is read off the far transmission formed above, whose refraction root
        // comes from the incidence sine directly and cannot land on the growing branch.
        //
        // With no tf anywhere the last event is a closed port and the result is zero.
        std::complex<double> leak_te(0.0, 0.0), leak_tm(0.0, 0.0);
        bool leak_first = false; // the wave entered the sheet as a leak
        if (produce && far_closed)
        {
            int it = is_scalar ? 4 : 1;
            std::complex<double> a_te, a_tm, b_te, b_tm;
            mat_near.interact_with(mat_slab, it, 0.0, fGHz, &a_te, &a_tm, nullptr, nullptr, nullptr, &leak_first,
                                   false, &ci, near_undeviated);
            if (leak_first)
                mat_slab.interact_with(mat_far, it, 0.0, fGHz, &b_te, &b_tm, nullptr, nullptr, nullptr, nullptr,
                                       false, &ci, true);
            else
            {
                mat_slab.interact_with(mat_far, it, 0.0, fGHz, &b_te, &b_tm, nullptr, nullptr, nullptr, nullptr,
                                       true, &cts_exact);
                if (is_scalar)
                    b_te = b_tm = std::polar(std::abs(b_te), std::arg(tf_out[0]));
            }
            leak_te = a_te * b_te, leak_tm = a_tm * b_tm;
        }

        if (produce && mat_slab.m > 0.0) // Incoherent route for mass-law slabs
        {
            // No series and no in-slab phase, but the two interface phases are genuine surface
            // effects and are kept. mass_min_path = 0.0 because a thin sheet is a real traversal;
            // the default floor exists to reject sub-millimeter paths in traced solids.
            //
            // The mass term is handed its two pieces separately: thk_normal is the surface-mass
            // surrogate and mass_c is the angle factor, so the sheet and a traced solid of
            // the same normal thickness form the same argument. The geometric-distance slot keeps
            // the product thk_normal * cos, which is what the minimum-path gate reads and what the
            // zero sentinel tests; neither the quotient thk / cos nor a floor on the cosine is
            // formed anywhere. At exact tangency the product is zero, which medium_gain would read
            // as "no geometric path supplied" and would then substitute the refracted path, so the
            // term is disabled outright in that case instead.
            //
            // Both pieces are real even where a complex incidence cosine was recovered. The
            // surrogate and its angle factor are lumped quantities the material table is calibrated
            // on, not wave quantities, and medium_gain takes them as real by construction; the same
            // reasoning excludes a mass-law slab from the exact in-medium branch in
            // SCALE_BY_MEDIUM. The two interface amplitudes above are wave quantities and do carry
            // the recovered cosine.
            //
            // The angle factor is the cosine the wave has in air. The plate is driven at the trace
            // wavenumber and the term is calibrated on a plate in air, so the angle is the one the
            // transverse invariant has there and not the one in a medium the wave was refracted
            // into on its way. Reached from air that is abs_cos_theta; a caller that reaches the
            // sheet from inside such a medium hands it in as mass_cos.
            const double mass_c = (mass_cos >= 0.0) ? mass_cos : abs_cos_theta;
            double mass_path = thk_normal * mass_c;
            double mass_floor = (mass_path > 0.0) ? 0.0 : 1.0e300;
            //
            // A sheet reached from inside a mass-law body is bonded to it: mass_bonded is the X0
            // that body's leg carries, and the sheet pays what the plate is worth above it.
            double gI = mat_slab.medium_gain(dist_slab, fGHz, abs_cos_theta_t, mass_path, 1.0, mass_floor,
                                             thk_normal, mass_c, mass_bonded);
            double aI = std::sqrt((gI < 0.0) ? 0.0 : gI);

            // A sheet in which the wave is evanescent is closed on both faces: lossless, and the
            // transverse invariant at or above its eps mu, the verdict Material::interact_with
            // gives a wave that meets it. Each face then passes the flat leak a bare face passes,
            // 1 - R' with the reflectance taken as one, whatever its Fresnel reflectance reads -
            // next to a lossy medium that is below one although the wave in the sheet carries no
            // flux, and rescaling a raw amplitude by (1 - R') / (1 - R0) there is a gain. A leak
            // is a statement about normal flux, so the two of them are the whole of the transmit
            // port and the near-to-far conversion, which belongs to an unbroken chain of raw
            // amplitudes, is not applied across the sheet. A face that is flat already, between
            // lossless media, keeps the value it had. The reflect port is not touched.
            //
            // A near medium that holds the wave undeviated does not take this block: the sheet
            // behind it is entered at an angle its Snell cosine does not describe. Towards the
            // lower index it takes the held branch below; towards the higher index it is left
            // on the composite, whose Snell cosine is real there.
            std::complex<double> w_n = (mat_near.eta(fGHz) + mat_near.eta_resonance(fGHz)) * mat_near.mu(fGHz);
            if (!far_closed && std::imag(eta_s_if * mu_s) == 0.0 && std::abs((w_n / (eta_s_if * mu_s)) * sin2_i) >= 1.0 &&
                !(near_undeviated && mat_near.m > 0.0))
            {
                double Rc_n = mat_slab.apply_tf_pair(mat_near, 1.0, fGHz), Rc_f = mat_slab.apply_tf_pair(mat_far, 1.0, fGHz);
                double a_n = std::sqrt((Rc_n < 1.0) ? 1.0 - Rc_n : 0.0), a_f = std::sqrt((Rc_f < 1.0) ? 1.0 - Rc_f : 0.0);
                if (!(1.0 - R0n_te < 1.0e-9))
                    tns_te = std::polar(a_n, std::arg(tn_fwd[0]));
                if (!(1.0 - R0n_tm < 1.0e-9))
                    tns_tm = std::polar(a_n, std::arg(tn_fwd[1]));
                if (!(1.0 - R0f_te < 1.0e-9))
                    tsf_te = std::polar(a_f, std::arg(tf_out[0]));
                if (!(1.0 - R0f_tm < 1.0e-9))
                    tsf_tm = std::polar(a_f, std::arg(tf_out[1]));
                fx_te = fx_tm = 1.0;
            }
            // A sheet behind a wave that is held undeviated in a mass-law medium, towards the lower
            // index: a later face of the same wall. The wave is a leak and keeps the incidence
            // cosine, so the sheet is the leak chain's own branch for a sheet entered by a leak:
            // near | sheet and sheet | far through Material::interact_with with the wave held,
            // each with the interface gain of the medium it enters, and the mass term. The near
            // face is then charged what the other direction charges, and the wall is one wall
            // from its two sides and the same wall with the sheet meshed as a traced solid. The
            // composite above would enter the sheet at the Snell cosine of two impedance
            // mappings, where both faces are flat leaks past the critical angle of the pair and
            // the near-to-far conversion, 1 / Re(z cos) of a mapping, multiplies them.
            //
            // The reflect port follows, as at a bare face: the reflection the other direction
            // sees at this cosine with its sign reversed, so the two ports of the near face add
            // up. The three gates are those of the undeviated rule, decided on the media and on
            // the direction the tracer holds.
            if (near_undeviated && mat_near.m > 0.0 && std::abs(w_n / (eta_s_if * mu_s)) > 1.0)
            {
                std::complex<double> h_te, h_tm;
                if (!far_closed)
                {
                    int it = is_scalar ? 4 : 1;
                    std::complex<double> a_te, a_tm, b_te, b_tm;
                    mat_near.interact_with(mat_slab, it, 0.0, fGHz, &a_te, &a_tm, nullptr, nullptr, nullptr, nullptr, false, &ci, true);
                    mat_slab.interact_with(mat_far, it, 0.0, fGHz, &b_te, &b_tm, nullptr, nullptr, nullptr, nullptr, false, &ci, true);
                    // The far medium's interface gain is owed only where it is newly entered
                    double un = (!far_is_air && mat_far.same_as(mat_near)) ? 1.0 / std::sqrt(mat_far.interface_gain(fGHz)) : 1.0;
                    h_te = (aI * un) * a_te * b_te, h_tm = (aI * un) * a_tm * b_tm;
                }
                else
                    h_te = aI * leak_te, h_tm = aI * leak_tm;
                std::complex<double> r_te, r_tm;
                mat_near.interact_with(mat_slab, is_scalar ? 3 : 0, 0.0, fGHz, &r_te, &r_tm, nullptr, nullptr, nullptr, nullptr, false, &ci, true);
                if (T_te)
                    *T_te = h_te;
                if (T_tm)
                    *T_tm = h_tm;
                if (R_te)
                    *R_te = r_te;
                if (R_tm)
                    *R_tm = r_tm;
                return true;
            }
            if (T_te)
                *T_te = far_closed ? aI * leak_te : ig * fx_te * aI * tns_te * tsf_te;
            if (T_tm)
                *T_tm = far_closed ? aI * leak_tm : ig * fx_tm * aI * tns_tm * tsf_tm;
            if (R_te) // Stokes reversal: the near interface seen from the incidence side
                *R_te = -rn_te;
            if (R_tm)
                *R_tm = -rn_tm;
            return true;
        }

        // One-way in-slab propagation factor phi, polarization-independent.
        //
        // Correction mode keeps the hybrid form: magnitude from medium_gain over the traced path
        // length, phase from the real index and the real projected cosine. Its angle comes from real
        // ray geometry and its result must not move.
        //
        // Produce mode takes both from one expression, exp(-j k0 n thk cos_theta_slab), with the index
        // and the cosine complex. Nothing propagates geometrically inside a sheet, so there is no real
        // direction the cosine has to be projected onto, and the incidence angle alone determines it.
        // The hybrid was the last term in the sheet's error budget once the interface amplitudes were
        // corrected: worth up to 0.039 dB at grazing incidence on a low-permittivity slab, and
        // removing it puts the composite on the analytic value at every angle.
        //
        // The conductivity loss lives inside n, so medium_gain must not supply it a second time; only
        // the alpha column is drawn from the material. The mass-law term cannot arrive here at all -
        // a mass-law slab returned through the incoherent route above. The index stays
        // resonance-excluded while the cosine does not, preserving the split the rest of this file
        // uses: a permittivity resonance is an interface effect, not a propagation one.
        std::complex<double> phi_1;
        if (produce)
        {
            std::complex<double> n_cx = std::sqrt(mat_slab.eta(fGHz) * mu_s); // complex, resonance-excluded
            double ex = mat_slab.excess_gain(dist_slab, fGHz);                // alpha only, no conductivity
            phi_1 = std::exp(std::complex<double>(0.0, -1.0) * (omega / c0) * n_cx * thk_normal * cts_exact) *
                    std::sqrt((ex < 0.0) ? 0.0 : ex);
        }
        else if (have_exact_ct)
        {
            // Same exact form as produce mode. The normal thickness is pure geometry - the traced path
            // times the traced cosine - and only the phase and loss across it use the recovered angle.
            std::complex<double> n_cx = std::sqrt(mat_slab.eta(fGHz) * mu_s); // resonance-excluded
            double ex = mat_slab.excess_gain(dist_slab, fGHz);                // alpha only
            phi_1 = std::exp(std::complex<double>(0.0, -1.0) * (omega / c0) * n_cx * thk_normal * cs) *
                    std::sqrt((ex < 0.0) ? 0.0 : ex);
        }
        else
        {
            double gL = mat_slab.medium_gain(dist_refract, fGHz, abs_cos_theta_t, dist_geo, abs_cos_theta);
            double abs_phi = std::sqrt((gL < 0.0) ? 0.0 : gL);
            double arg_phi = -(omega / c0) * n_cur * dist_refract * abs_cos_theta_t * abs_cos_theta_t;
            phi_1 = std::polar(abs_phi, arg_phi);
        }

        std::complex<double> phi2 = phi_1 * phi_1;

        std::complex<double> denom_te = std::complex<double>(1.0, 0.0) - rn_te * rf_te * phi2;
        std::complex<double> denom_tm = std::complex<double>(1.0, 0.0) - rn_tm * rf_tm * phi2;

        if (!produce)
        {
            // Survival gate (stronger polarization) + near-pole clamp (either) -> re-emit
            double g2L = mat_slab.medium_gain(2.0 * dist_refract, fGHz, abs_cos_theta_t, 2.0 * dist_geo, abs_cos_theta);
            g2L = (g2L < 0.0) ? 0.0 : g2L;
            double rr_te = (Rn_te * Rf_te < 0.0) ? 0.0 : Rn_te * Rf_te;
            double rr_tm = (Rn_tm * Rf_tm < 0.0) ? 0.0 : Rn_tm * Rf_tm;
            double rho = std::sqrt(((rr_te > rr_tm) ? rr_te : rr_tm) * g2L);
            if (!verdict_given && (rho < eps || std::abs(denom_te) < 1.0e-2 || std::abs(denom_tm) < 1.0e-2))
                return false;

            // A verdict that is given cannot be handed back: the bounces are not followed. A
            // near-pole denominator is clamped in magnitude and keeps its phase, as for a sheet.
            if (verdict_given)
                for (std::complex<double> *d : {&denom_te, &denom_tm})
                    if (std::abs(*d) < 1.0e-2)
                        *d = std::polar(1.0e-2, std::arg(*d));
        }
        else
        {
            // Neither bail-out has a sheet equivalent: there is no interior to re-emit into, so a
            // near-pole denominator is clamped in magnitude and keeps its phase, and the weak-slab
            // gate is dropped - a weak slab simply resolves close to its single-pass value.
            auto CLAMP_POLE = [](std::complex<double> &d)
            {
                if (std::abs(d) < 1.0e-2)
                    d = std::polar(1.0e-2, std::arg(d));
            };
            CLAMP_POLE(denom_te), CLAMP_POLE(denom_tm);

            std::complex<double> phi = phi_1;

            // Displacement reference for the transmit port.
            //
            // A sheet has zero geometric thickness, so it stands for a real layer that OCCUPIES thk of
            // whatever medium it sits in. The tracer walked that length as that medium, and the rest of
            // the path already carries it, so it has to come back out here or the same length is
            // counted twice. Without this a 10 cm pane modelled as a quad comes out 10 cm of optical
            // path too long, which is exactly the k0*thk*cos(theta_i) offset it was measured at.
            //
            // WHICH medium is the one the sheet physically sits in, not the one it is attached to. For
            // a glued pair that is the side which is not the solid: an absorber on a concrete wall
            // occupies room air, so entering the room-air side it is the near medium, and leaving
            // through the wall it is the far one. An isolated sheet has near == far and the choice does
            // not arise.
            //
            // HOW MUCH is capped by disp_budget, because taking a medium back out is a division - by
            // its loss as much as its phase - and a sheet thicker than the host it sits in would
            // amplify rather than attenuate. Past the cap the sheet still carries its full thickness
            // and only the replacement is limited, so the path simply comes out longer than the
            // geometry. That is the right answer for a laminate whose layers do not fit inside their
            // host, and it keeps every material crossing intact.
            //
            // phi2 in the denominator is a round trip inside the slab and stays absolute. The
            // reflect port is handed to a ray that turns around at the mesh plane, so it owes the
            // round trip through the part of the layer that lies in front of that plane: half of
            // it where one medium is on both sides, the mesh plane being the mid-plane of the
            // layer; all of it for a glued entry, whose layer lies in the near medium in front of
            // the solid; none for a glued exit, whose layer lies beyond.
            const Material &disp_mtl = disp_far ? mat_far : mat_near;
            std::complex<double> eta_d = disp_mtl.eta(fGHz) + disp_mtl.eta_resonance(fGHz);
            std::complex<double> mu_d = disp_mtl.mu(fGHz);
            std::complex<double> n_d = std::sqrt(disp_mtl.eta(fGHz) * mu_d); // resonance-excluded, as above
            std::complex<double> cd_arg = 1.0 - ((mat_near.eta(fGHz) + mat_near.eta_resonance(fGHz)) *
                                                 mat_near.mu(fGHz) / (eta_d * mu_d)) *
                                                    sin2_i;
            if (qd_evanescent(cd_arg))
                cd_arg = std::complex<double>(std::real(cd_arg), -0.0);
            std::complex<double> c_disp = std::sqrt(cd_arg);

            double thk_n = thk_normal; // sheet thickness along the normal
            double t_disp = (disp_budget >= 0.0 && disp_budget < thk_n) ? disp_budget : thk_n;
            if (t_disp < 0.0)
                t_disp = 0.0;

            std::complex<double> phi_disp =
                std::exp(std::complex<double>(0.0, -1.0) * (omega / c0) * n_d * t_disp * c_disp);
            std::complex<double> phi_t = (std::abs(phi_disp) > 0.0) ? phi / phi_disp : phi;
            std::complex<double> undisp = (std::abs(phi_disp) > 0.0) ? std::complex<double>(1.0, 0.0) / phi_disp
                                                                     : std::complex<double>(1.0, 0.0);
            std::complex<double> front = mat_far.same_as(mat_near) ? undisp
                                                                   : (disp_far ? std::complex<double>(1.0, 0.0) : undisp * undisp);

            // The leak is undeviated, so the host length it displaces was walked at the incidence
            // cosine. On the far side of a closed pair the Snell cosine above is evanescent, and
            // dividing by that decay would amplify a wave that never took it.
            //
            // The sheet's own factor follows the wave. One that entered by an ordinary crossing is a
            // Fresnel wave inside the sheet: it takes phi as it stands, and what the far face
            // reflects is summed with the series the reflect port carries, because a sheet has no
            // interior the tracer could follow a bounce into and the energy would otherwise be
            // dropped. One that entered as a leak kept the incidence cosine, where the Snell cosine
            // is evanescent and phi would charge a decay it never took; it takes the form an
            // undeviated traversal of a traced solid takes, the loss along the path and the phase of
            // the real index on the thickness axis. Its bounces have no phase the media define, so
            // it carries no series, as the mass-law route carries none.
            std::complex<double> series_te(1.0, 0.0), series_tm(1.0, 0.0);
            if (far_closed)
            {
                if (leak_first)
                {
                    double c_re = std::abs(std::real(ci));
                    double d_leak = (c_re > 0.0) ? thk_n / c_re : dist_max;
                    if (!(d_leak < dist_max))
                        d_leak = dist_max;
                    double gL = mat_slab.medium_gain(d_leak, fGHz, 1.0, 0.0, 0.0, 1.0e300); // mass term off
                    phi = std::polar(std::sqrt((gL < 0.0) ? 0.0 : gL), -(omega / c0) * n_cur * thk_n * c_re);
                }
                else
                    series_te = 1.0 / denom_te, series_tm = 1.0 / denom_tm;

                std::complex<double> phi_dl =
                    std::exp(std::complex<double>(0.0, -1.0) * (omega / c0) * n_d * t_disp * ci);
                phi_t = (std::abs(phi_dl) > 0.0) ? phi / phi_dl : phi;
            }

            if (T_te)
                *T_te = far_closed ? leak_te * phi_t * series_te : ig * fx_te * tns_te * tsf_te * phi_t / denom_te;
            if (T_tm)
                *T_tm = far_closed ? leak_tm * phi_t * series_tm : ig * fx_tm * tns_tm * tsf_tm * phi_t / denom_tm;
            if (R_te) // front-face reflection plus the series leaking back out through the near face
                *R_te = front * (-rn_te + ig_echo * tns_te * tsn_te * rf_te * phi2 / denom_te);
            if (R_tm)
                *R_tm = front * (-rn_tm + ig_echo * tns_tm * tsn_tm * rf_tm * phi2 / denom_tm);
            return true;
        }

        // Correction mode returns the series factor and nothing else. The port the caller holds
        // already carries the per-channel Fresnel coefficient of its own interface, so S is the whole
        // of what is owed here.
        //
        // This used to be S * sqrt(Tsp * corr), with corr = 1 / (Tn_avg * Tf_avg), on the reasoning
        // that Material::interact_with hands over an AVERAGED two-interface magnitude that has to be
        // converted back to a per-channel one. Both halves of that reasoning are wrong, and the pair
        // was measured emitting exactly |T_X|^2 * |S_X|^2 * Tsp_X - the per-channel transmittance
        // applied a second time on top of a port that already carried it.
        //
        // The averaging in interact_with never distorted a channel ratio, and after the energy
        // convention change there is no averaged rescale left in the transmission branch at all:
        // each channel carries its own normal-flux obliquity factor, and the TE/TM average enters
        // only through tf, which is polarization-agnostic by design. And even if a per-channel
        // conversion were needed here, corr divides by a TWO-interface single pass while
        // ray_state_update corrects one interaction at a time, so the port in hand carries one
        // interface, not two.
        //
        // The two branches now agree except for the scalar duplication: scalar mode carries one port
        // and the caller scales a stored xprmat in place rather than rebuilding it from cTE, so both
        // outputs get the same factor.
        std::complex<double> S_te = std::complex<double>(1.0, 0.0) / denom_te;
        std::complex<double> S_tm = is_scalar ? S_te : std::complex<double>(1.0, 0.0) / denom_tm;

        if (T_te)
            *T_te = S_te;
        if (T_tm)
            *T_tm = S_tm;
        return true;
    }
}

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# medium_gain
Linear gain of a ray traversing a homogeneous lossy medium

- Computes `g = 10^(-A/10)`, where `A` [dB] is the total attenuation accumulated over a path
  of length `dist` inside the medium. The loss combines three contributions:
  - Conductivity-based loss from the complex permittivity model of ITU-R P.2040-1: `ε_r = a·(f/fRef)^b`,
    `σ = c·(f/fRef)^d`. These give an attenuation length `Δ` and a per-meter power loss `8.686 / Δ` dB/m.
  - Distance absorption of the form `α·(f/fRef)^αB` dB/m, intended to model excess loss not captured
    by `σ` (e.g. foliage, scattering media).
  - An acoustic mass-law term `A = m·log10((f/fRef)·dist)` dB, added once (not per meter) when `m > 0`,
    `dist` exceeds ~1.5 mm, and the term is positive; `m` is the mass-law slope in dB/decade. At oblique
    incidence the term carries the mass law's angle shape, `A·log10(1 + x²)/log10(1 + X₀²)` with
    `X₀ = 10^(A/20)` and `x = X₀·cos θ`, which is `A` at normal incidence, tapers to zero at tangency,
    and has an angle slope of 20 dB/decade of `cos θ` independent of `m`. This entry point supplies
    `cos θ = 1` and therefore returns `A` unchanged.
- The penetration-loss columns (`att`, `attB`) of `mtl_prop` are not used — they describe
  thin-slab transmission loss, not propagation through a finite-thickness medium.

## Declaration:
```
dtype quadriga_lib::medium_gain(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype dist,
    dtype center_frequency);
```

## Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`
- **`iM`** —  1-based material index (0 = no material / air)
- **`dist`** — Path length of the ray inside the medium
- **`center_frequency`** — Center frequency in [Hz]

## Returns:
- Linear in-medium gain in `[0, 1]`; multiply by the incident field/power gain to get the value after the medium

## See also:
- [[ray_mesh_interact]] (for complex ray-material interactions)
- [[obj_file_read]] (defines mtl_prop format)
MD!*/

// ISSUE: medium_gain currently only evaluates one distance. However, for media using mass-law, it requires
// two different inputs, one for the refracted path feeding the medium attenuation and one for the
// geometric path feeding mass. This then needs to propagate to the call site, currently calc_diffraction_gain.

template <typename dtype>
dtype quadriga_lib::medium_gain(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
                                arma::uword iM, dtype dist, dtype center_frequency)
{
    if (!std::isfinite((double)center_frequency) || center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    MaterialCols<dtype> cols(mtl_prop); // validates column lengths and physical sanity
    if (iM > cols.n_mtl)
        throw std::invalid_argument("Material index out of bound.");
    return (dtype)Material(cols, iM).medium_gain((double)dist, (double)center_frequency * 1e-9);
}

template float quadriga_lib::medium_gain(const std::unordered_map<std::string, std::vector<float>> &mtl_prop, arma::uword iM, float dist, float center_frequency);
template double quadriga_lib::medium_gain(const std::unordered_map<std::string, std::vector<double>> &mtl_prop, arma::uword iM, double dist, double center_frequency);

/*!MD
# interface_gain
Linear gain of a wave crossing a thin interface (lumped penetration loss)

- Computes `g = 10^(-A/10)`, where `A` [dB] is the lumped transmission loss applied once when a
  ray enters a material (the air-to-material or material-to-material front-side crossing). It is
  independent of path length and is applied on top of the Fresnel interface term `1 - abs(R)²`:
  - Power-law penetration loss `att·(f/fRef)^attB` (e.g. 3GPP TR 38.901 building-entry loss).
  - An optional Lorentzian coincidence feature `coiA / (1 + (coiQ·(f - coiF)/coiF)²)`, active only
    when `coiF > 0` and `coiA != 0`; negative `coiA` is a transmission dip (acoustic coincidence),
    positive `coiA` a stop-band. The total is clamped to `>= 0`.
- The reflection / conductivity columns (`a`, `b`, `c`, `d`) and the in-medium columns
  (`alpha`, `alphaB`, `m`) of `mtl_prop` are not used here — the Fresnel reflection is handled by
  the caller and the distance-dependent loss by [[medium_gain]].

## Declaration:
```
dtype quadriga_lib::interface_gain(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype center_frequency);
```

## Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`
- **`iM`** — 1-based material index (0 = no material / air)
- **`center_frequency`** — Center frequency in [Hz]

## Returns:
- Linear interface gain in `[0, 1]`; multiply by the incident field/power gain to get the value after the interface

## See also:
- [[medium_gain]] (for the distance-dependent in-medium loss)
- [[ray_mesh_interact]] (for complex ray-material interactions)
- [[obj_file_read]] (defines mtl_prop format)
MD!*/

template <typename dtype>
dtype quadriga_lib::interface_gain(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
                                   arma::uword iM, dtype center_frequency)
{
    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    MaterialCols<dtype> cols(mtl_prop);
    if (iM > cols.n_mtl)
        throw std::invalid_argument("Material index out of bound.");
    return (dtype)Material(cols, iM).interface_gain((double)center_frequency * 1e-9);
}

template float quadriga_lib::interface_gain(const std::unordered_map<std::string, std::vector<float>> &mtl_prop, arma::uword iM, float center_frequency);
template double quadriga_lib::interface_gain(const std::unordered_map<std::string, std::vector<double>> &mtl_prop, arma::uword iM, double center_frequency);

/*!MD
# refractive_index
Real refractive index of a homogeneous medium

- Returns `n = Re(sqrt(ε_r · μ_r))`, the real part of the complex refractive index, using the
  ITU-R P.2040-1 permittivity model `ε_r = a·(f/fRef)^b` together with the relative permeability `μ_r`.
- Only the bulk (base) permittivity is used. The coincidence / resonance features
  (`coiF`, `coiQ`, `coiA`, `resF`, ...) are excluded, since they model a thin-interface surface
  effect, not bulk propagation, and must not enter the geometric refraction index.
- Air (`iM = 0`) returns `1`.

## Declaration:
```
dtype quadriga_lib::refractive_index(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype center_frequency);
```

## Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`
- **`iM`** — 1-based material index (0 = no material / air)
- **`center_frequency`** — Center frequency in [Hz]

## Returns:
- Real refractive index of the medium relative to air

## See also:
- [[medium_gain]] (for the distance-dependent in-medium loss)
- [[ray_mesh_interact]] (for complex ray-material interactions)
- [[obj_file_read]] (defines mtl_prop format)
MD!*/

template <typename dtype>
dtype quadriga_lib::refractive_index(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
                                     arma::uword iM, dtype center_frequency)
{
    if (!std::isfinite((double)center_frequency) || center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    MaterialCols<dtype> cols(mtl_prop); // validates column lengths and physical sanity
    if (iM > cols.n_mtl)
        throw std::invalid_argument("Material index out of bound.");
    Material M(cols, iM);
    double fGHz = (double)center_frequency * 1e-9;
    return (dtype)std::real(std::sqrt(M.eta(fGHz) * M.mu(fGHz))); // base eta, no resonance term
}

template float quadriga_lib::refractive_index(const std::unordered_map<std::string, std::vector<float>> &mtl_prop, arma::uword iM, float center_frequency);
template double quadriga_lib::refractive_index(const std::unordered_map<std::string, std::vector<double>> &mtl_prop, arma::uword iM, double center_frequency);

/*!MD
# ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

- Computes interaction of plane waves with planar interfaces between homogeneous isotropic media.
- Supports beam-based modeling via triangular ray tubes (`trivec`, `tridir`).
- Face side determined by vertex order; CCW winding = front, CW = back (right-hand rule); front-side hit with
  FBS≠SBS → air-to-media; back-side hit with FBS≠SBS → media-to-air; FBS=SBS with opposing normals → media-to-media.
- With `compact = true` (default), rays with `fbs_ind = 0` (no interaction) are omitted from output,
  so `n_rayN ≤ n_ray`; with `compact = false` they are kept as transparent pass-throughs (`n_rayN = n_ray`).
- Output direction encoding (spherical/Cartesian) matches input `tridir` format.
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves).
- Thin sheets: a material with `thk > 0` marks its faces as a **sheet** — a single face carrying an
  authored thickness instead of a traced interior. Each crossing is one atomic entry + traversal + exit,
  with the internal multi-bounce series resolved in closed form. `thk = 0`, or no `thk` column, keeps the
  classic behavior where thickness comes from the geometry and the [[ray_state_update]] state machine.
  Modeling notes:
  - Two coincident sheets of the *same* material are collapsed to a single crossing. Exporters emit
    back-to-back face pairs to make geometry double-sided, and treating such a pair as a two-layer
    laminate would double both the thickness and the loss. Author a genuine laminate as two *different*
    materials, or as one material with the summed `thk`.
  - Stack depth is capped at two faces, because the FBS/SBS intersector returns two. A third layer is
    hit as a separate FBS after the relaunch and concatenates incoherently, and `solid + sheet + solid`
    is already three faces. Scenes have to be designed around this.
  - A closed solid built from a sheet material is legal, not an error: it becomes a hollow box, with
    sheet loss on entry, the embedding medium inside, and sheet loss on exit.
- Types 3–5 (scalar) use a single TE-only coefficient (acoustic simulation with impedance-mapped
  materials, `ε` derived from `Z`); total internal reflection is handled as in the EM path
  (`snell·sinθ ≥ 1` into a lossless medium ⇒ unit reflection, refraction collapses to undeviated transmission;
  a lossy medium has no critical angle).
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

## Declaration:
```
void quadriga_lib::ray_mesh_interact(
    int interaction_type,
    dtype center_frequency,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *mesh,
    const arma::uvec *mtl_ind,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::u32_vec *fbs_ind,
    const arma::u32_vec *sbs_ind,
    const arma::Mat<dtype> *trivec = nullptr,
    const arma::Mat<dtype> *tridir = nullptr,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    arma::Mat<dtype> *fbsN = nullptr,
    arma::Mat<dtype> *sbsN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *trivecN = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    arma::Col<dtype> *fbs_angleN = nullptr,
    arma::Col<dtype> *thicknessN = nullptr,
    arma::Col<dtype> *edge_lengthN = nullptr,
    arma::Mat<dtype> *normal_vecN = nullptr,
    std::vector<uint8_t> *out_typeN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,
    bool compact = false,
    arma::u32_vec *ray_indN = nullptr);
```

## Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission, 5 = scalar refraction
- **`center_frequency`** — Center frequency in [Hz]
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see [[obj_file_read]]; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`.
  0 = face has no material (air). NULL → all faces treated as air.
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]);
  each value has length `n_mtl`. NULL → air defaults used.
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian
- **`compact`** *(optional)* — If `true`, no-hit rays are dropped and `n_rayN ≤ n_ray`. If
  `false` (default), all rays are kept (`n_rayN = n_ray`) and no-hit rays are written as a transparent pass-through
  (gain 1, identity `xprmat`, `out_type = 0`).

## Outputs:
- **`origN`** — New origins after interaction, nudged off the face along the travel direction by ~8 float ULP
  at the interaction-point coordinate magnitude; `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`fbsN`**, **`sbsN`** — First/second interaction points in GCS; `[n_ray, 3]`
- **`gainN`** — Linear power gain of the interaction. Derived from `xprmatN`, not an independent factor; `[n_rayN]`
- **`xprmatN`** — Polarization transfer matrix; not normalized: the entries are amplitude coefficients
  that already carry the interaction gain. For types 0–2: interleaved complex, col-major
  `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`, i.e. the Jones matrix `M = [VV VH; HV HH]` acting as
  `e_out = M · e_in`; the incoming V/H basis is built from the incident direction, the outgoing basis from
  `path_dirN` (`eH = ẑ × k̂` normalized, `eV = eH × k̂`); `[8, n_rayN]`. For types 3–5 (scalar): `[Re Im]`
  where `Re + jIm` is the scalar pressure coefficient; `[2, n_rayN]`.
  - *Included:* TE/TM interface coefficients (with the lumped `interface_gain` folded in for the transmission
    classes 1/2/4/5) and the incidence-plane orientation.
  - *Excluded:* in-medium attenuation and excess phase — added by [[ray_state_update]] — and FSPL / spreading
    loss, which is never applied here or downstream.
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if inputs not provided
- **`fbs_angleN`** — Incidence angle at FBS in rad; `[n_rayN]`
- **`thicknessN`** — Geometric FBS-to-SBS distance; `[n_rayN]`. Unrelated to the material `thk` column:
  this is what the mesh measured, `thk` is what the material table authored.
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (∞ if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code, bit-encoded (`qd::bits<uint8_t>`); `[n_rayN]`<br><br>
   |  Bit | Meaning                                                                 |
   | :--: | ----------------------------------------------------------------------- |
   |   0  | OK flag (0 = no valid interaction / undefined)                          |
   |   1  | Front-side flag — meaning depends on bits 2 and 6, see below            |
   |   2  | Co-located FBS/SBS flag (1 = single point, required for media-to-media) |
   |   3  | Same-direction flag (FBS and SBS normals point the same way)            |
   |   4  | Corner-hit flag (FBS/SBS faces not parallel)                            |
   |   5  | Total-reflection flag (also set when a transmission factor forced it)   |
   |   6  | Thin-sheet flag (at least one of FBS/SBS is a `thk > 0` material)       |
   Bit 1 by context:<br><br>
   | Context                                     | Bit 1 = 1 means                               |
   | ------------------------------------------- | --------------------------------------------- |
   | Solid (bit 6 = 0)                           | Front-side hit on FBS (o→i, or M2 hit first)  |
   | Isolated sheet (bit 6 = 1, bit 2 = 0)       | Front-side hit on FBS; informational only     |
   | Sheet glued to a solid (exactly one sheet)  | Front-side hit on the **solid** face          |
   | Two co-located sheets                       | FBS traversed first (0 = SBS traversed first) |
   Reachable composite values (add 32 for the total-reflection variant):<br><br>
   | Code  |  TIR  | Description                                                                        |
   | :---: | :---: | ---------------------------------------------------------------------------------- |
   |   0   |   —   | No hit                                                                             |
   |   1   |  33   | Single hit, inside→outside (exit)                                                  |
   |   3   |  35   | Single hit, outside→inside (entry)                                                 |
   |   5   |  37   | Media-to-media, M1 (current, back) hit first                                       |
   |   7   |  39   | Media-to-media, M2 (next, front) hit first                                         |
   |  13   |  45   | Overlapping faces, inside-inside→outside                                           |
   |  15   |  47   | Overlapping faces, outside→inside-inside                                           |
   |  21   |  53   | Corner hit, inside→outside→inside                                                  |
   |  23   |  55   | Corner hit, outside→inside→outside                                                 |
   |  29   |  61   | Corner hit, inside-inside→outside                                                  |
   |  31   |  63   | Corner hit, outside→inside-inside                                                  |
   |  65   |   —   | Isolated sheet, back-side hit; or a sheet at a corner with SBS dropped             |
   |  67   |   —   | Isolated sheet, front-side hit; or a sheet at a corner with SBS dropped            |
   |  69   |  101  | Glued exit (solid→sheet→air); or a stack with SBS first; or a collapsed duplicate  |
   |  71   |  103  | Glued entry (air→sheet→solid); or a stack with FBS first; or a collapsed duplicate |
   |  79   |   —   | Two-sheet stack with an ill-posed order (geometric FBS→SBS fallback was used)      |
   Codes 69 and 71 are shared by three cases, deliberately: [[ray_state_update]] builds its own material
   set and holds both `mtl_ind_fbs` and `mtl_ind_sbs`, so it can re-derive sheet-ness per face and material
   identity on the pair. The full case is `(composite code, is_sheet(FBS), is_sheet(SBS), same material)`.
   Bit 6 alone is a hint meaning *at least one face is a sheet*; the re-derivation is authoritative and
   [[ray_state_update]] performs it. Total reflection can only co-occur with bit 6 for the glued cases,
   where the media either side of the sheet differ. Bit 7 is unused.<br><br>
   The near / far media written into a sheet composite here are **provisional**. They come from face
   geometry alone, so air is assumed on any side the mesh does not name, which is only correct while the
   ray is in air. [[ray_state_update]] holds the tracked medium and re-resolves the composite against the
   real bounding pair whenever medium tracking is enabled; without it, the air-bounded result stands.
- **`path_dirN`** — Refraction-correct path direction: mirror for types 0/3, Snell direction for types 1/2/4/5; `[n_rayN, 3]`.
  For undeviated transmission (types 1/4) this is the *refracted* direction, which differs from the geometric continuation
  (along the incoming ray) used for `origN`/`destN`; it lets downstream code recover the true transmission angle.
  Out of a mass-law medium (`m > 0`) there is no bend with any type: the wave is held undeviated at every face.
- **`ray_indN`** — 0-based input ray index for each output ray (inverse of the internal compaction map; order-preserving); `[n_rayN]`

## See also:
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-State Machine</a> (companion document)
- [[obj_file_read]] (for loading `mesh` and `mtl_prop` from OBJ file)
- [[ray_state_update]] (inside/outside state machine)
- [[icosphere]] (for generating beams)
- [[ray_triangle_intersect]] (for computing FBS and SBS positions)
- [[ray_point_intersect]] (for calculating beam interactions with sampling points)
MD!*/

template <typename dtype>
void quadriga_lib::ray_mesh_interact(int interaction_type,
                                     dtype center_frequency,
                                     const arma::Mat<dtype> *orig,
                                     const arma::Mat<dtype> *dest,
                                     const arma::Mat<dtype> *mesh,
                                     const arma::uvec *mtl_ind,
                                     const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
                                     const arma::u32_vec *fbs_ind,
                                     const arma::u32_vec *sbs_ind,
                                     const arma::Mat<dtype> *trivec,
                                     const arma::Mat<dtype> *tridir,
                                     arma::Mat<dtype> *origN,
                                     arma::Mat<dtype> *destN,
                                     arma::Mat<dtype> *fbsN,
                                     arma::Mat<dtype> *sbsN,
                                     arma::Col<dtype> *gainN,
                                     arma::Mat<dtype> *xprmatN,
                                     arma::Mat<dtype> *trivecN,
                                     arma::Mat<dtype> *tridirN,
                                     arma::Col<dtype> *fbs_angleN,
                                     arma::Col<dtype> *thicknessN,
                                     arma::Col<dtype> *edge_lengthN,
                                     arma::Mat<dtype> *normal_vecN,
                                     std::vector<uint8_t> *out_typeN,
                                     arma::Mat<dtype> *path_dirN,
                                     bool compact,
                                     arma::u32_vec *ray_indN)
{
    // ISSUE: standalone callers with no state tracking get no in-medium loss (before they did);
    // resolve by adding state tracking to the engine or charging medium loss separately

    if (interaction_type < 0 || interaction_type > 5)
        throw std::invalid_argument("Interaction type must be either (0) EM Reflection, (1) EM Transmission, (2) EM Refraction, (3) Scalar Reflection, (4) Scalar Transmission, (5) Scalar Refraction");
    bool is_scalar = interaction_type >= 3;
    int geometry_type = interaction_type % 3;

    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    double fGHz = (double)center_frequency * 1.0e-9;

    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    const arma::uword n_ray = orig->n_rows; // Number of rays
    const dtype *p_orig = orig->memptr();

    if (dest == nullptr)
        throw std::invalid_argument("Input 'dest' cannot be NULL.");
    if (dest->n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");
    if (dest->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");
    const dtype *p_dest = dest->memptr();

    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    const arma::uword n_mesh = mesh->n_rows; // Number of mesh elements
    const dtype *p_mesh = mesh->memptr();

    if (fbs_ind == nullptr)
        throw std::invalid_argument("Input 'fbs_ind' cannot be NULL.");
    if (fbs_ind->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'fbs_ind' does not match number of rows in 'orig'.");
    const unsigned *p_fbs_ind = fbs_ind->memptr();

    if (sbs_ind == nullptr)
        throw std::invalid_argument("Input 'sbs_ind' cannot be NULL.");
    if (sbs_ind->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'sbs_ind' does not match number of rows in 'orig'.");

    const unsigned *p_sbs_ind = sbs_ind->memptr();

    if (mtl_ind && !mtl_ind->is_empty() && mtl_ind->n_elem != n_mesh)
        throw std::invalid_argument("Length of 'mtl_ind' must match the number of mesh faces.");

    // Check input data for ray tube
    int use_ray_tube = 0;
    if (trivec && !trivec->is_empty())
    {
        if (tridir == nullptr || tridir->is_empty())
            throw std::invalid_argument("In order to use ray tubes, both 'trivec' and 'tridir' must be given.");
        if (trivec->n_cols != 9)
            throw std::invalid_argument("Input 'trivec' must have 9 columns.");
        if (trivec->n_rows != n_ray)
            throw std::invalid_argument("Number of rows in 'orig' and 'trivec' dont match.");
        if (tridir->n_cols != 6 && tridir->n_cols != 9)
            throw std::invalid_argument("Input 'tridir' must have 6 or 9 columns.");
        if (tridir->n_rows != n_ray)
            throw std::invalid_argument("Number of rows in 'orig' and 'tridir' dont match.");
        use_ray_tube = (tridir->n_cols == 6) ? 1 : 2;
    }
    else if (tridir && !tridir->is_empty())
        throw std::invalid_argument("In order to use ray tubes, both 'trivec' and 'tridir' must be given.");
    const dtype *p_trivec = trivec ? trivec->memptr() : nullptr;
    const dtype *p_tridir = tridir ? tridir->memptr() : nullptr;

    // Resolve material columns once; air (empty) table when no material model is supplied.
    const arma::uword *p_mtl_ind = (mtl_ind == nullptr || mtl_ind->is_empty()) ? nullptr : mtl_ind->memptr();
    MaterialCols<dtype> MaterialSet = (mtl_prop) ? MaterialCols<dtype>(*mtl_prop) : MaterialCols<dtype>();
    if (p_mtl_ind && mtl_ind->max() > MaterialSet.n_mtl)
        throw std::invalid_argument("Entries of 'mtl_ind' cannot exceed the number of materials.");

    // Get number of output rays and build output ray index
    // - Only consider rays that have at least one interaction with the mesh, i.e. 'fbs_ind != 0'
    // - Without compaction the map is the identity, so it is not materialized at all: at 1e8 rays
    //   the index array alone costs 400 MB, and the range check then parallelizes.
    //   With compaction the running counter is a prefix sum and the loop has to stay serial.
    unsigned n_rayN_u = 0;
    arma::u32_vec output_ray_index; // 1-based, empty = identity map

    if (compact)
    {
        output_ray_index.set_size(n_ray); // not zero-filled: every element is written below
        unsigned *p_ind = output_ray_index.memptr();
        for (size_t i_ray = 0; i_ray < n_ray; ++i_ray) // Ray loop
        {
            if (p_fbs_ind[i_ray] > n_mesh) // Invalid, must be 1 ... n_mesh (1-based index)
                throw std::invalid_argument("Some values in 'fbs_ind' exceed number of mesh elements.");
            if (p_sbs_ind[i_ray] > n_mesh)
                throw std::invalid_argument("Some values in 'sbs_ind' exceed number of mesh elements.");

            if (p_fbs_ind[i_ray] == 0)
                p_ind[i_ray] = 0; // drop no-hits
            else
                p_ind[i_ray] = ++n_rayN_u; // keep
        }
    }
    else // Identity map: only the index range is left to check, and that is a reduction
    {
        int bad = 0;
#pragma omp parallel for schedule(static) reduction(| : bad) if (n_ray >= 51200)
        for (long long i_ray = 0; i_ray < (long long)n_ray; ++i_ray) // Ray loop
        {
            if (p_fbs_ind[i_ray] > n_mesh) // Invalid, must be 1 ... n_mesh (1-based index)
                bad |= 1;
            if (p_sbs_ind[i_ray] > n_mesh)
                bad |= 2;
        }

        if (bad & 1)
            throw std::invalid_argument("Some values in 'fbs_ind' exceed number of mesh elements.");
        if (bad & 2)
            throw std::invalid_argument("Some values in 'sbs_ind' exceed number of mesh elements.");

        n_rayN_u = (unsigned)n_ray;
    }
    const arma::uword n_rayN = (arma::uword)n_rayN_u;
    const unsigned *p_ray_map = output_ray_index.empty() ? nullptr : output_ray_index.memptr();

    // Allocate output memory, if needed
    if (origN && (origN->n_rows != n_rayN || origN->n_cols != 3))
        origN->set_size(n_rayN, 3);
    dtype *p_origN = origN ? origN->memptr() : nullptr;

    if (destN && (destN->n_rows != n_rayN || destN->n_cols != 3))
        destN->set_size(n_rayN, 3);
    dtype *p_destN = destN ? destN->memptr() : nullptr;

    if (fbsN && (fbsN->n_rows != n_rayN || fbsN->n_cols != 3))
        fbsN->set_size(n_rayN, 3);
    dtype *p_fbsN = fbsN ? fbsN->memptr() : nullptr;

    if (sbsN && (sbsN->n_rows != n_rayN || sbsN->n_cols != 3))
        sbsN->set_size(n_rayN, 3);
    dtype *p_sbsN = sbsN ? sbsN->memptr() : nullptr;

    if (gainN && gainN->n_elem != n_rayN)
        gainN->set_size(n_rayN);
    dtype *p_gainN = gainN ? gainN->memptr() : nullptr;

    const arma::uword nXPR = is_scalar ? 2 : 8; // NUmber of columns in xprmat (8 for EM, 2 for scalar)
    if (xprmatN && (xprmatN->n_rows != nXPR || xprmatN->n_cols != n_rayN))
        xprmatN->set_size(nXPR, n_rayN);
    dtype *p_xprmatN = xprmatN ? xprmatN->memptr() : nullptr;

    if (trivecN && use_ray_tube && (trivecN->n_rows != n_rayN || trivecN->n_cols != 9))
        trivecN->set_size(n_rayN, 9);
    else if (trivecN && !use_ray_tube && !trivecN->is_empty())
        trivecN->reset();
    dtype *p_trivecN = trivecN ? trivecN->memptr() : nullptr;

    if (tridirN && use_ray_tube == 1 && (tridirN->n_rows != n_rayN || tridirN->n_cols != 6))
        tridirN->set_size(n_rayN, 6);
    else if (tridirN && use_ray_tube == 2 && (tridirN->n_rows != n_rayN || tridirN->n_cols != 9))
        tridirN->set_size(n_rayN, 9);
    else if (tridirN && !use_ray_tube && !tridirN->is_empty())
        tridirN->reset();
    dtype *p_tridirN = tridirN ? tridirN->memptr() : nullptr;

    if (fbs_angleN && fbs_angleN->n_elem != n_rayN)
        fbs_angleN->set_size(n_rayN);
    dtype *p_fbs_angleN = fbs_angleN ? fbs_angleN->memptr() : nullptr;

    if (thicknessN && thicknessN->n_elem != n_rayN)
        thicknessN->set_size(n_rayN);
    dtype *p_thicknessN = thicknessN ? thicknessN->memptr() : nullptr;

    if (edge_lengthN && edge_lengthN->n_elem != n_rayN)
        edge_lengthN->set_size(n_rayN);
    dtype *p_edge_lengthN = edge_lengthN ? edge_lengthN->memptr() : nullptr;

    if (normal_vecN && (normal_vecN->n_rows != n_rayN || normal_vecN->n_cols != 6))
        normal_vecN->set_size(n_rayN, 6);
    dtype *p_normal_vecN = normal_vecN ? normal_vecN->memptr() : nullptr;

    if (out_typeN && out_typeN->size() != n_rayN) // out_typeN is a std::vector, not arma::Col
        out_typeN->resize(n_rayN, 0);
    uint8_t *p_out_typeN = out_typeN ? out_typeN->data() : nullptr;

    if (path_dirN && (path_dirN->n_rows != n_rayN || path_dirN->n_cols != 3))
        path_dirN->set_size(n_rayN, 3);
    dtype *p_path_dirN = path_dirN ? path_dirN->memptr() : nullptr;

    if (ray_indN && ray_indN->n_elem != n_rayN)
        ray_indN->set_size(n_rayN);
    unsigned *p_ray_indN = ray_indN ? ray_indN->memptr() : nullptr;

    // Under refraction (geometry_type 2) the per-vertex incidence can escalate to whole-tube TIR below,
    // which changes cTE/cTM, the path direction and the TIR flag. Skip the tube only when no tube
    // output is requested AND the tube cannot affect the center-ray result.
    if (use_ray_tube && geometry_type != 2 && p_trivecN == nullptr && p_tridirN == nullptr && p_edge_lengthN == nullptr)
        use_ray_tube = 0;

#pragma omp parallel for schedule(static)
    for (long long i_ray = 0; i_ray < (long long)n_ray; ++i_ray) // Ray loop
    {
        size_t iRx = (size_t)i_ray;                                   // Ray x-index
        size_t iRy = iRx + n_ray;                                     // Ray y-index
        size_t iRz = iRy + n_ray;                                     // Ray z-index
        size_t i_rayN = p_ray_map ? (size_t)p_ray_map[iRx] - 1 : iRx; // Output ray index, 0-based

        // Normalization lambda
        auto NORMALIZE = [](double &x, double &y, double &z, bool apply = true) -> double
        {
            double len = std::sqrt(x * x + y * y + z * z);
            if (apply && len > 2e-7)
            {
                double scl = 1.0 / len;
                x *= scl, y *= scl, z *= scl;
            }
            else if (apply) // Fallback
                x = 1.0, y = 0.0, z = 0.0;
            return len;
        };

        auto LENGTH = [](double x, double y, double z) -> double
        { return std::sqrt(x * x + y * y + z * z); };

        auto SET1 = [i_rayN](dtype *ptr, double val)
        {
            if (ptr)
                ptr[i_rayN] = (dtype)val;
        };

        auto SET3 = [i_rayN, n_rayN](dtype *ptr, double x, double y, double z)
        {
            if (ptr)
                ptr[i_rayN] = (dtype)x, ptr[i_rayN + n_rayN] = (dtype)y, ptr[i_rayN + 2 * n_rayN] = (dtype)z;
        };

        auto SETL = [i_rayN, n_rayN](dtype *ptr, double *data, size_t L = 9, bool set_row = true)
        {
            if (ptr)
            {
                if (set_row) // Set row
                    for (size_t l = 0; l < L; ++l)
                        ptr[i_rayN + l * n_rayN] = (dtype)data[l];
                else // Set column
                    for (size_t l = 0; l < L; ++l)
                        ptr[i_rayN * L + l] = (dtype)data[l];
            }
        };

        if (compact && p_fbs_ind[i_ray] == 0) // Compact stream, skip non-hits
            continue;

        double Ox = (double)p_orig[iRx], Oy = (double)p_orig[iRy], Oz = (double)p_orig[iRz]; // Origin position
        double Dx = (double)p_dest[iRx], Dy = (double)p_dest[iRy], Dz = (double)p_dest[iRz]; // Destination position
        double ODx = Dx - Ox, ODy = Dy - Oy, ODz = Dz - Oz;                                  // Ray direction O to D

        // Shift D back 2 ULP to drop paths that end on the FBS face
        double odScale = std::max({std::abs(ODx), std::abs(ODy), std::abs(ODz), 1e-30});
        double posScale = std::max({std::abs(Dx), std::abs(Dy), std::abs(Dz), 1.0});
        double offset = 2.0 * posScale * 1.1920929e-7 / odScale;
        Dx -= offset * ODx, Dy -= offset * ODy, Dz -= offset * ODz;

        ODx = Dx - Ox, ODy = Dy - Oy, ODz = Dz - Oz; // Update direction O to D
        double OD_length = NORMALIZE(ODx, ODy, ODz);

        size_t iFBS = (size_t)p_fbs_ind[iRx]; // Mesh FBS index, 1-based
        size_t iSBS = (size_t)p_sbs_ind[iRx]; // Mesh SBS index, 1-based

        if (iFBS == 0) // no mesh hit
        {
            SET3(p_origN, Dx, Dy, Dz);
            SET3(p_destN, Dx + colocation_dist * ODx, Dy + colocation_dist * ODy, Dz + colocation_dist * ODz);
            SET3(p_fbsN, Dx, Dy, Dz);
            SET3(p_sbsN, Dx, Dy, Dz);
            SET1(p_gainN, 1.0);

            double xprmat[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
            SETL(p_xprmatN, xprmat, nXPR, false);

            if (use_ray_tube)
            {
                double tv[9] = {0.0}, td[9] = {0.0};
                double edge_len = 0.0;

                for (size_t iTube = 0; iTube < 3; ++iTube)
                {
                    // Vertex origin = ray origin + trivec offset
                    size_t io = iRx + 3 * iTube * n_ray;
                    size_t ii = 3 * iTube;

                    double Tx = Ox + (double)p_trivec[io];
                    double Ty = Oy + (double)p_trivec[io + n_ray];
                    double Tz = Oz + (double)p_trivec[io + 2 * n_ray];

                    // Vertex direction from tridir (unchanged by a transparent pass)
                    double Vx, Vy, Vz, az = 0.0, el = 0.0;
                    if (use_ray_tube == 1) // spherical
                    {
                        az = (double)p_tridir[iRx + 2 * iTube * n_ray];
                        el = (double)p_tridir[iRx + (2 * iTube + 1) * n_ray];
                        double c = std::cos(el);
                        Vx = std::cos(az) * c, Vy = std::sin(az) * c, Vz = std::sin(el);

                        size_t ij = 2 * iTube;
                        td[ij] = az, td[ij + 1] = el;
                    }
                    else // cartesian
                    {
                        Vx = (double)p_tridir[io], Vy = (double)p_tridir[io + n_ray], Vz = (double)p_tridir[io + 2 * n_ray];
                        NORMALIZE(Vx, Vy, Vz);
                        td[ii] = Vx, td[ii + 1] = Vy, td[ii + 2] = Vz;
                    }

                    // Advance the vertex ray to the wavefront plane through D with normal OD = (Nx,Ny,Nz)
                    double denom = Vx * ODx + Vy * ODy + Vz * ODz;
                    double Wx, Wy, Wz;
                    if (std::abs(denom) < 1e-6) // vertex parallel to the wavefront plane
                    {
                        edge_len = INFINITY;
                        Wx = Tx, Wy = Ty, Wz = Tz; // sane fallback instead of NaN
                    }
                    else
                    {
                        double d = ((Dx - Tx) * ODx + (Dy - Ty) * ODy + (Dz - Tz) * ODz) / denom;
                        if (d < 0.0 || d > 1.0e5)
                            edge_len = INFINITY;
                        Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;
                    }
                    tv[ii] = Wx - Dx, tv[ii + 1] = Wy - Dy, tv[ii + 2] = Wz - Dz; // trivec relative to origN = D
                }
                SETL(p_trivecN, tv);
                if (use_ray_tube == 1) // Spherical
                    SETL(p_tridirN, td, 6);
                else // Cartesian
                    SETL(p_tridirN, td);

                if (p_edge_lengthN)
                {
                    if (!std::isinf(edge_len))
                    {
                        double e = 0.0, dx, dy, dz, l;
                        dx = tv[3] - tv[0], dy = tv[4] - tv[1], dz = tv[5] - tv[2];
                        l = dx * dx + dy * dy + dz * dz;
                        e = l > e ? l : e;
                        dx = tv[6] - tv[0], dy = tv[7] - tv[1], dz = tv[8] - tv[2];
                        l = dx * dx + dy * dy + dz * dz;
                        e = l > e ? l : e;
                        dx = tv[6] - tv[3], dy = tv[7] - tv[4], dz = tv[8] - tv[5];
                        l = dx * dx + dy * dy + dz * dz;
                        e = l > e ? l : e;
                        edge_len = std::sqrt(e);
                    }
                    p_edge_lengthN[i_rayN] = (dtype)edge_len;
                }
            }

            SET1(p_fbs_angleN, 1.570796326794897);
            SET1(p_thicknessN, 0.0);

            if (p_normal_vecN)
                SET3(p_normal_vecN, ODx, ODy, ODz), SET3(&p_normal_vecN[3 * n_rayN], ODx, ODy, ODz);

            if (p_out_typeN)
                p_out_typeN[i_rayN] = 0;

            SET3(p_path_dirN, ODx, ODy, ODz);

            if (p_ray_indN)
                p_ray_indN[i_rayN] = (unsigned)iRx;

            continue;
        } // end iFBS == 0 (no mesh hit)

        // Material indices for FBS and SBS faces (0 if no material table)
        arma::uword iMF = (p_mtl_ind && iFBS) ? p_mtl_ind[iFBS - 1] : 0;
        arma::uword iMS = (p_mtl_ind && iSBS) ? p_mtl_ind[iSBS - 1] : 0;

        // Thin-sheet flags for FBS and SBS
        bool sheetFBS = MaterialSet.is_sheet(iMF);
        bool sheetSBS = MaterialSet.is_sheet(iMS);

        // Compute the FBS intersect point, initialize with fallback
        double Fx = (double)p_mesh[iFBS - 1];
        double Fy = (double)p_mesh[iFBS - 1 + n_mesh];
        double Fz = (double)p_mesh[iFBS - 1 + 2 * n_mesh];
        double Nx = 0.0, Ny = 0.0, Nz = 0.0; // FBS normal vector
        double cos_theta = 0.0;              // Angle between FBS normal and incoming ray
        double OF_length = 0.0;              // Distance from origin to FBS
        if (iFBS)
        {
            double E1x = (double)p_mesh[iFBS - 1 + 3 * n_mesh] - Fx,
                   E1y = (double)p_mesh[iFBS - 1 + 4 * n_mesh] - Fy,
                   E1z = (double)p_mesh[iFBS - 1 + 5 * n_mesh] - Fz;
            double E2x = (double)p_mesh[iFBS - 1 + 6 * n_mesh] - Fx,
                   E2y = (double)p_mesh[iFBS - 1 + 7 * n_mesh] - Fy,
                   E2z = (double)p_mesh[iFBS - 1 + 8 * n_mesh] - Fz;

            // The longest triangle edge used to be measured here, as the in-plane extent driving the
            // thin-sheet grazing clamp. It is gone with the clamp. Nothing in the sheet path reads a
            // property of the triangulation any more, so a face and the same face subdivided produce
            // the same coefficients.

            // Plane normal vector
            Nx = E1y * E2z - E1z * E2y, Ny = E1z * E2x - E1x * E2z, Nz = E1x * E2y - E1y * E2x;
            NORMALIZE(Nx, Ny, Nz);

            // Ray-plane intersection
            cos_theta = ODx * Nx + ODy * Ny + ODz * Nz;         // goes to zero as the ray approaches tangency
            if (OD_length > 2e-7 && std::abs(cos_theta) > 2e-7) // guard degenerate ray and grazing/parallel plane
            {
                OF_length = ((Fx - Ox) * Nx + (Fy - Oy) * Ny + (Fz - Oz) * Nz) / cos_theta;
                if (OF_length <= 0.0) // Origin lies in FBS plane (include)
                    Fx = Ox, Fy = Oy, Fz = Oz, OF_length = 0.0;
                else if (OF_length < OD_length) // True FBS intersect (include)
                    Fx = Ox + OF_length * ODx, Fy = Oy + OF_length * ODy, Fz = Oz + OF_length * ODz;
                else // Destination lies in FBS plane (exclude)
                    Fx = Dx, Fy = Dy, Fz = Dz, OF_length = OD_length;
            }
        }

        // Calculate incidence angle between surface of the mesh element at FBS and incoming ray
        cos_theta = (cos_theta < -1.0) ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta); // Boundary fix
        double thetaF = std::acos(cos_theta) - 1.570796326794897;                    // Angle between FBS face and incoming ray

        // Compute the SBS intersect point, initialize with fallback or destination
        double Sx = iSBS ? (double)p_mesh[iSBS - 1] : Dx;
        double Sy = iSBS ? (double)p_mesh[iSBS - 1 + n_mesh] : Dy;
        double Sz = iSBS ? (double)p_mesh[iSBS - 1 + 2 * n_mesh] : Dz;
        double Mx = ODx, My = ODy, Mz = ODz; // SBS normal vector
        double thetaS = 0.0;                 // Angle between SBS face and incoming ray
        if (iSBS)
        {
            double E1x = (double)p_mesh[iSBS - 1 + 3 * n_mesh] - Sx,
                   E1y = (double)p_mesh[iSBS - 1 + 4 * n_mesh] - Sy,
                   E1z = (double)p_mesh[iSBS - 1 + 5 * n_mesh] - Sz;
            double E2x = (double)p_mesh[iSBS - 1 + 6 * n_mesh] - Sx,
                   E2y = (double)p_mesh[iSBS - 1 + 7 * n_mesh] - Sy,
                   E2z = (double)p_mesh[iSBS - 1 + 8 * n_mesh] - Sz;

            // Plane normal vector
            Mx = E1y * E2z - E1z * E2y, My = E1z * E2x - E1x * E2z, Mz = E1x * E2y - E1y * E2x;
            NORMALIZE(Mx, My, Mz);

            // Ray-plane intersection
            thetaS = ODx * Mx + ODy * My + ODz * Mz;         // = cos(thetaS)
            if (OD_length > 2e-7 && std::abs(thetaS) > 2e-7) // guard degenerate ray and grazing/parallel plane
            {
                double OS_length = ((Sx - Ox) * Mx + (Sy - Oy) * My + (Sz - Oz) * Mz) / thetaS;
                if (OS_length <= OF_length) // SBS before FBS (not allowed)
                    Sx = Fx, Sy = Fy, Sz = Fz, OS_length = OF_length;
                else if (OS_length < OD_length) // True SBS intersect
                    Sx = Ox + OS_length * ODx, Sy = Oy + OS_length * ODy, Sz = Oz + OS_length * ODz;
                else // Destination lies in SBS plane
                    Sx = Dx, Sy = Dy, Sz = Dz, OS_length = OD_length;
            }

            // Incidence angle
            thetaS = (thetaS < -1.0) ? -1.0 : (thetaS > 1.0 ? 1.0 : thetaS);
            thetaS = std::acos(thetaS) - 1.570796326794897;
        }
        double FS_length = LENGTH(Sx - Fx, Sy - Fy, Sz - Fz); // Length of vector FS

        // Determine the type of the interaction
        bool material_to_material = false; // Assume no material to material transition

        // Set type codes
        qd::bits<uint8_t> out_flags = OF_length == OD_length ? 0 : 1;    // Set OK flag out_flags[0] = 1
        out_flags[1] = thetaF >= 0.0;                                    // Set front-side flag
        bool colocated_faces = FS_length < colocation_dist && iSBS != 0; // Two colocated faces
        out_flags[2] = colocated_faces;

        // Thin-sheet case per the classification below, 0 = no sheet involved
        int sheet_case = 0;

        // Effective front/back verdict. For a solid, and for a sheet on its own, this is just the FBS
        // front-side test. It diverges only where the FBS normal is not the one that decides: a sheet
        // has no inside, so its own orientation carries no information, and for a stack the question is
        // which layer comes first rather than which side of a face was hit.
        bool front_eff = out_flags[1];

        bool normals_opposing = false, normals_equal = false;
        if (colocated_faces)
        {
            const double lim = 1.0e-4;
            normals_opposing = std::abs(Nx + Mx) < lim && std::abs(Ny + My) < lim && std::abs(Nz + Mz) < lim;
            normals_equal = std::abs(Nx - Mx) < lim && std::abs(Ny - My) < lim && std::abs(Nz - Mz) < lim;

            if (normals_opposing)
                material_to_material = true; // Opposing normal vectors, Material to material transition
            else if (normals_equal)
                out_flags[3] = true; // Equal normal vectors = overlapping or duplicate faces
            else                     // FBS/SBS faces not parallel
            {
                out_flags[4] = 1; // Corner hit flag
                if ((out_flags[1] && thetaS >= 0.0) ||
                    (!out_flags[1] && thetaS < 0.0))
                    out_flags[3] = true; // Same direction flag
            }
        }

        // Thin-sheet classification. Every branch below is gated on a sheet being present, so a scene
        // without a thk column, or with all thk zero, reaches none of them and keeps the bits above.
        if (colocated_faces && (sheetFBS || sheetSBS))
        {
            if (!normals_opposing && !normals_equal)
            {
                // Corner with a sheet. A sheet meeting a face at an angle is not a slab: the two faces
                // share no interior and there is no well-defined traversal. Drop SBS and handle FBS on
                // its own, as a sheet when FBS is the sheet and as an ordinary single hit otherwise.
                // The corner and co-location bits are cleared so downstream sees the single hit it is
                // actually being handed; the relaunch still goes past SBS, because that is driven by the
                // local colocated_faces flag rather than by the bit.
                sheet_case = 6;
                out_flags[2] = false;
                out_flags[3] = false;
                out_flags[4] = false;
                out_flags[6] = sheetFBS;
            }
            else if (sheetFBS && sheetSBS)
            {
                if (Material(MaterialSet, iMF).same_as(Material(MaterialSet, iMS)))
                {
                    // Collapsed duplicate. Two coincident sheets of identical material are an exporter
                    // double-sided artifact, not a laminate - a real laminate would double thk instead.
                    // Collapse regardless of normal parity, so matched-normal duplicates are caught too.
                    // Without this the stack rule below would charge every double-sided export twice and
                    // add interference between a layer and its own copy. Since the pair is inside
                    // colocation_dist there is no air gap between them, so nothing real is lost.
                    sheet_case = 5;
                    out_flags[3] = false;
                    out_flags[6] = true;
                }
                else
                {
                    // Ordered stack of two different sheets, a coated window being the motivating case:
                    // it has to give different answers from either side. The layer hit on its front side
                    // is traversed first. Normals point outward from the assembly, so this is the only
                    // ordering information an exporter carries, and it is reversal-symmetric - reverse
                    // the ray and the layer order reverses with it. When both faces are front-hit or
                    // both back-hit the order is ill-posed; fall back to the geometric order FBS->SBS,
                    // which is correct whenever the faces have any real separation, and record in bit 3
                    // that the order was inferred from geometry rather than from normals.
                    sheet_case = 4;
                    bool frontF = thetaF >= 0.0, frontS = thetaS >= 0.0;
                    front_eff = (frontF != frontS) ? frontF : true;
                    out_flags[3] = frontF == frontS;
                    out_flags[6] = true;
                }
            }
            else
            {
                // Sheet glued to a solid face. The sheet always sits on the incidence side of that solid
                // face: a sheet glued to the solid's other boundary would be co-located with that face
                // instead of this one, unless the solid is thinner than colocation_dist, which is
                // degenerate. So the traversal order follows from co-location alone and needs no normal
                // comparison, which is why either parity is accepted here. Wallpaper on the room side of
                // a wall: entering from the room is air|paper|wall, leaving the wall is wall|paper|air.
                // The front/back verdict has to come from the solid face, since the sheet has none.
                front_eff = sheetFBS ? (thetaS >= 0.0) : (thetaF >= 0.0);
                sheet_case = front_eff ? 2 : 3; // 2 = glued entry (o->i), 3 = glued exit (i->o)
                out_flags[3] = false;
                out_flags[6] = true;
            }
        }
        else if (sheetFBS)
        {
            // Isolated sheet: no second face within colocation_dist, so the same medium sits on both
            // sides. Here that medium is air; a sheet embedded in a solid needs the current medium and
            // is ray_state_update's problem, not this function's.
            sheet_case = 1;
            out_flags[6] = true;
        }

        // Bit 1 carries the effective verdict, not the raw FBS test. For solids and isolated sheets
        // front_eff is the FBS test and this write changes nothing.
        out_flags[1] = front_eff;

        // Flip normal vector in case of back side illumination
        // This is pure geometry: it orients N toward the incidence side so that qd_reflect and
        // qd_refract bend the right way. It therefore stays on the FBS test even for a glued sheet
        // whose own normal happens to be reversed relative to the solid it sits on - front_eff decides
        // which material is near and which is far, never which way a normal points.
        if (thetaF < 0.0)
            Nx = -Nx, Ny = -Ny, Nz = -Nz,
            cos_theta = ODx * Nx + ODy * Ny + ODz * Nz,
            cos_theta = (cos_theta < -1.0) ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta);

        // Limit value to 0 ... 1 for calculating reflection and transmission coefficients
        double abs_cos_theta = std::abs(cos_theta);

        // Incidence-side (medium 1) and entered/reflected-off (medium 2) materials, default air.
        // 1-based: Material(cols, 0) -> air. M2 always carries the FBS-face material whose
        // interface_gain is the transition gain (front: iMF; back: air, or iMS for M2M).
        // A sheet crossing does not fit that pattern: Material::interact_with is a two-medium interface
        // function, while a sheet is three media, two interfaces and one slab. So M1 and M2 become the
        // media on either side of the sheet - near and far - and the sheet itself goes into M_slab.
        // material_to_material plays no part in the sheet branches; the pair is one atomic crossing,
        // not a transition between two traced media.
        Material M1, M2, M_slab;
        bool sheet_resolve = false; // coefficients come from qd_slab_resolve instead of interact_with
        if (sheet_case == 1 || sheet_case == 5)
        {
            // Isolated sheet, or a collapsed double-sided duplicate: the same medium sits on both
            // sides, so near = far = air. The sheet material never reaches interact_with - it enters
            // only through the slab resolve, which is what makes its normal orientation irrelevant.
            M_slab = Material(MaterialSet, iMF);
            sheet_resolve = true;
        }
        else if (sheet_case == 2 || sheet_case == 3)
        {
            // Glued pair: air | sheet | solid on entry, solid | sheet | air on exit. Which of the two
            // co-located faces carries the sheet does not matter, because front_eff already took the
            // verdict from the solid face, so FBS-sheet and SBS-sheet geometry land here identically.
            // A back-side hit on the solid implies the ray is already inside it, which is the same
            // trick the M1/M2 block below uses to recover the incidence medium without any state.
            M_slab = Material(MaterialSet, sheetFBS ? iMF : iMS);
            Material M_solid(MaterialSet, sheetFBS ? iMS : iMF);
            if (front_eff) // glued entry, o->i
                M2 = M_solid;
            else // glued exit, i->o
                M1 = M_solid;
            sheet_resolve = true;
        }
        else if (thetaF >= 0.0) // front hit: entered material = FBS face (iMF)
        {
            M2 = Material(MaterialSet, iMF);
            if (material_to_material) // SBS (front) hit first
                M1 = Material(MaterialSet, iMS);
        }
        else // back hit: incidence material = FBS face (iMF)
        {
            M1 = Material(MaterialSet, iMF);
            if (material_to_material) // FBS (back) hit first, entered = SBS
                M2 = Material(MaterialSet, iMS);
        }

        // Interface evaluation. For a solid this one call supplies both the coefficients and the
        // geometry. For a sheet the two are split by purpose: the geometry comes from the near/far pair
        // evaluated here, the coefficients from the slab resolve below.
        std::complex<double> cTE, cTM, cos_theta2, eta1_div_eta2;
        double eta; // Snell ratio sqrt|eta1*mu1 / eta2*mu2|
        bool total_reflection;
        // The incidence angle here is the geometric one, so the wave is held undeviated by
        // construction; ray_state_update replaces the result wherever a refracted direction exists.
        M1.interact_with(M2, interaction_type, thetaF, fGHz, &cTE, &cTM, &cos_theta2, &eta1_div_eta2, &eta, &total_reflection,
                         false, nullptr, true);

        // Direction and total reflection come from near/far, never from near/slab. Snell's law is
        // transitive - n_near*sin_near = n_slab*sin_slab = n_far*sin_far - so the net bend and the
        // critical angle depend only on the two outer media and the sheet cannot affect the outgoing
        // direction at all. Reading them off interact_with(near, sheet) would invent total reflection
        // past the air|sheet critical angle, and in scalar mode the impedance mapping readily gives
        // n < 1, which would turn an acoustic sheet into a mirror.
        if (sheet_case == 1 || sheet_case == 5)
            total_reflection = false;        // near == far: unreachable except at exact tangency, so force it off
        bool tir_central = total_reflection; // pre-ray-tube TIR state

        auto SHEET_COMPOSITE = [&](bool far_closed)
        {
            double thk = M_slab.thk;

            // In-slab propagation angle. Only cos_theta2 is used; the coefficients and the TIR verdict
            // of this near|sheet interface are deliberately thrown away for the reason just given.
            std::complex<double> ct2_slab;
            M1.interact_with(M_slab, interaction_type, thetaF, fGHz, nullptr, nullptr, &ct2_slab, nullptr, nullptr, nullptr);

            // A thin sheet is a laterally infinite layer, which is what the rest of the composite
            // already assumes: plane-wave Fresnel boundaries and a plane-parallel internal series.
            // There is no grazing clamp.
            //
            // There used to be one, flooring both cosines at thk / sqrt(thk^2 + ext^2) on the argument
            // that a grazing ray through a finite pane would leave through its side. Three things were
            // wrong with it. It floored the INCIDENCE cosine, which sits upstream of both Fresnel
            // boundaries and of the exact in-slab cosine, so it froze the entire interface at the
            // onset angle rather than capping a traversal: measured, transmission held 28.8 dB above
            // the analytic slab at 89.9 deg while the specular mirror that grazing incidence should
            // produce was removed. It keyed the onset to the longest edge of the hit triangle, so the
            // same wall answered differently depending on how finely it was meshed - a 1000 m pane as
            // two triangles clamps past 89.9998 deg, the same pane in 1 m triangles past 89.757 deg,
            // and at 89.99 deg the two differ by 27.7 dB. And it was inert in production anyway, since
            // ray_state_update re-resolves every sheet crossing and discards what is computed here.
            //
            // Finite lateral extent is an edge-diffraction question. It is not expressible from a
            // triangle index in any case: the extent that matters belongs to the connected coplanar
            // region, and the mesh carries no adjacency. At the angles where the walk-off would
            // matter it is also comparable to the Fresnel zone over which the interaction is
            // averaged, so a per-ray walk-off test resolves geometry below the scale on which the
            // transfer function is defined.
            //
            // Nothing diverges without it: qd_slab_resolve is written in the normal thickness and the
            // complex in-slab cosine, and forms the one quotient that can grow under a loss-saturation
            // cap of its own.
            double ct_geo = abs_cos_theta;
            double ct_slab = std::abs(std::real(ct2_slab));

            // Displacement side and budget. The sheet sits on whichever side of a glued pair is not
            // the solid, which is the far side only for a glued exit. The budget is what this
            // function can see of the medium being displaced - the leg it has just travelled - and it
            // only matters when that medium is lossy. ray_state_update redoes this with the tracked
            // state and a two-sided budget whenever medium tracking is on; this is the provisional
            // value for the untracked path.
            bool disp_far = (sheet_case == 3);
            const Material &M_disp = disp_far ? M2 : M1;
            double disp_budget = M_disp.same_as(Material()) ? -1.0 : 0.8 * OF_length * ct_geo;

            // Thickness enters only here. Produce mode takes the normal thickness and the two
            // cosines and owns every length it derives from them; the geometric-path slot is unused.
            std::complex<double> Tte, Ttm, Rte, Rtm;
            qd_slab_resolve(M1, M_slab, M2, thk, 0.0, ct_slab, ct_geo,
                            fGHz, is_scalar, 0.0, true, &Tte, &Ttm, &Rte, &Rtm,
                            disp_far, disp_budget, std::complex<double>(0.0, 0.0),
                            std::complex<double>(0.0, 0.0), nullptr, far_closed, true);

            // Both ports come out of the one pass over the same Fresnel quantities. The reflection port
            // is its own Airy sum, not what interact_with would return for a half-space: expect real
            // angular and frequency structure instead of a smooth Fresnel value.
            if (geometry_type == 0) // Reflection
                cTE = Rte, cTM = Rtm;
            else // Transmission or refraction
                cTE = Tte, cTM = Ttm;
        };
        if (sheet_resolve)
            SHEET_COMPOSITE(total_reflection);

        // Calculate the center path direction after medium interaction (normalized to length 1)
        double FDx = Dx - Fx, FDy = Dy - Fy, FDz = Dz - Fz; // Vector from FBS to destination
        double FD_length = NORMALIZE(FDx, FDy, FDz);        // Length of path from FBS to destination

        // Per-vertex incidence, computed once and reused by the tube loop below
        double Vedge[9];                     // incoming vertex directions
        double cos_thetaV[4], sin_thetaV[4]; // incidence cosine/sine per leg
        if (use_ray_tube)
        {
            cos_thetaV[0] = abs_cos_theta; // spine == center ray
            sin_thetaV[0] = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta);
            for (int iTube = 0; iTube < 3; ++iTube)
            {
                double Vx, Vy, Vz;
                if (use_ray_tube == 1) // Spherical: az/el -> direction
                {
                    double az = (double)p_tridir[iRx + 2 * iTube * n_ray];
                    double el = (double)p_tridir[iRx + (2 * iTube + 1) * n_ray];
                    double c = std::cos(el);
                    Vx = std::cos(az) * c, Vy = std::sin(az) * c, Vz = std::sin(el);
                }
                else // Cartesian
                {
                    size_t o = iRx + 3 * iTube * n_ray;
                    Vx = (double)p_tridir[o], Vy = (double)p_tridir[o + n_ray], Vz = (double)p_tridir[o + 2 * n_ray];
                    NORMALIZE(Vx, Vy, Vz);
                }
                Vedge[3 * iTube] = Vx, Vedge[3 * iTube + 1] = Vy, Vedge[3 * iTube + 2] = Vz;
                double c = std::abs(Vx * Nx + Vy * Ny + Vz * Nz);
                c = (c > 1.0) ? 1.0 : c;
                cos_thetaV[iTube + 1] = c, sin_thetaV[iTube + 1] = std::sqrt(1.0 - c * c);
            }

            // Whole-tube TIR: pass through (undeviated) if the spine OR any edge is past critical.
            // Skipped where near == far, since the Snell ratio is then 1 and the test can only trip on
            // numerical tangency - which would escalate the entire tube into a mirror on the strength
            // of a rounding artifact at the spine. Skipped where the far medium is lossy as well: it
            // has no critical angle, for an edge as little as for the spine.
            if (geometry_type == 2 && sheet_case != 1 && sheet_case != 4 && sheet_case != 5 &&
                std::imag((M2.eta(fGHz) + M2.eta_resonance(fGHz)) * M2.mu(fGHz)) == 0.0)
                for (int i = 0; i <= 3 && !total_reflection; ++i)
                    if (eta * sin_thetaV[i] >= 1.0)
                        total_reflection = true;
        }

        // Outgoing spine direction. An isolated sheet and a collapsed duplicate need no special case:
        // with near == far the interface call returned eta = 1 and cos_theta2 = cos_theta, so the
        // refraction branch reproduces the incoming direction. An ordered stack does: its interface
        // call saw the legacy pair of its two layers, which is not near | far. It has one medium on
        // both sides, so the spine, the tube and path_dirN keep the incoming direction.
        const bool no_bend = sheet_case == 4;
        if (geometry_type == 0) // Reflection, normalized by default
            qd_reflect(ODx, ODy, ODz, Nx, Ny, Nz, FDx, FDy, FDz);
        else if (geometry_type == 1 || total_reflection || no_bend) // Transmission without refraction
            FDx = ODx, FDy = ODy, FDz = ODz;                        // New path direction = same as incoming ray, already normalized
        else                                                        // Refraction
            qd_refract(ODx, ODy, ODz, Nx, Ny, Nz, eta, abs_cos_theta, cos_theta2, FDx, FDy, FDz);

        // Update origin and direction of the ray tube vertices
        double p_trivec_tmp[9] = {};
        double p_tridir_tmp[9] = {};
        double edge_length_tmp = 0.0;
        if (use_ray_tube)
        {
            // Process each vertex-ray separately
            for (int iTube = 0; iTube < 3; ++iTube)
            {
                // Load origin and direction
                double Tx = Ox, Ty = Oy, Tz = Oz, az = 0.0, el = 0.0;
                double Vx = Vedge[3 * iTube], Vy = Vedge[3 * iTube + 1], Vz = Vedge[3 * iTube + 2];
                if (iTube == 0)
                    Tx += (double)p_trivec[iRx], Ty += (double)p_trivec[iRy], Tz += (double)p_trivec[iRz];
                else if (iTube == 1)
                    Tx += (double)p_trivec[iRx + 3 * n_ray], Ty += (double)p_trivec[iRx + 4 * n_ray], Tz += (double)p_trivec[iRx + 5 * n_ray];
                else // iTube == 2
                    Tx += (double)p_trivec[iRx + 6 * n_ray], Ty += (double)p_trivec[iRx + 7 * n_ray], Tz += (double)p_trivec[iRx + 8 * n_ray];

                // Calculate intersect point of the vertex-ray with the face
                double denom = Vx * Nx + Vy * Ny + Vz * Nz;
                bool no_usable_hit = std::abs(denom) < 1e-6; // true => parallel, no face intersection
                double d = no_usable_hit ? 0.0 : ((Fx - Tx) * Nx + (Fy - Ty) * Ny + (Fz - Tz) * Nz) / denom;
                double Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;
                no_usable_hit = no_usable_hit || d > 1.0e5 || d < 0.0;

                if (no_usable_hit) // no usable face intersection
                    edge_length_tmp = INFINITY;

                if (geometry_type == 0) // Reflection
                {
                    if (no_usable_hit) // Use orthogonal projection on vertex ray
                    {
                        d = ((Fx - Tx) * Vx + (Fy - Ty) * Vy + (Fz - Tz) * Vz) / (Vx * Vx + Vy * Vy + Vz * Vz);
                        Tx = Tx + Vx * d - Fx, Ty = Ty + Vy * d - Fy, Tz = Tz + Vz * d - Fz; // Scaled vertex - updates T
                        double a = 2.0 * (Tx * Nx + Ty * Ny + Tz * Nz);                      // Reflection of T on face
                        Tx -= a * Nx, Ty -= a * Ny, Tz -= a * Nz;
                    }
                    else // Use intersect point W as new vertex origin
                        Tx = Wx - Fx, Ty = Wy - Fy, Tz = Wz - Fz;

                    // Update vertex direction
                    qd_reflect(Vx, Vy, Vz, Nx, Ny, Nz, Vx, Vy, Vz);
                    if (use_ray_tube == 1)
                    {
                        Vz = (Vz < -1.0) ? -1.0 : (Vz > 1.0 ? 1.0 : Vz); // Boundary fix
                        az = std::atan2(Vy, Vx), el = std::asin(Vz);
                    }
                }
                else // Transmission and Refraction
                {
                    if (no_usable_hit) // Use orthogonal projection on vertex ray
                    {
                        d = ((Fx - Tx) * Vx + (Fy - Ty) * Vy + (Fz - Tz) * Vz) / (Vx * Vx + Vy * Vy + Vz * Vz);
                        Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;
                    }

                    // Update ray tube coordinates
                    Tx = Wx - Fx, Ty = Wy - Fy, Tz = Wz - Fz;

                    // Vertex ray directions remains the same for Transmission
                    if (geometry_type == 2 && !total_reflection && !no_bend) // Refraction (skipped when the whole tube passes through under TIR)
                    {
                        std::complex<double> cos_theta2V = std::sqrt(1.0 - eta1_div_eta2 * sin_thetaV[iTube + 1] * sin_thetaV[iTube + 1]);
                        qd_refract(Vx, Vy, Vz, Nx, Ny, Nz, eta, cos_thetaV[iTube + 1], cos_theta2V, Vx, Vy, Vz);
                    }
                }

                // Spherical output: derive az/el from the final vertex direction
                if (use_ray_tube == 1)
                {
                    Vz = (Vz < -1.0) ? -1.0 : (Vz > 1.0 ? 1.0 : Vz); // Boundary fix
                    az = std::atan2(Vy, Vx), el = std::asin(Vz);
                }

                // Write new vertex ray origin and direction - convert back to dtype
                if (iTube == 0)
                {
                    p_trivec_tmp[0] = Tx, p_trivec_tmp[1] = Ty, p_trivec_tmp[2] = Tz;
                    if (use_ray_tube == 1)
                        p_tridir_tmp[0] = az, p_tridir_tmp[1] = el;
                    else
                        p_tridir_tmp[0] = Vx, p_tridir_tmp[1] = Vy, p_tridir_tmp[2] = Vz;
                }
                else if (iTube == 1)
                {
                    p_trivec_tmp[3] = Tx, p_trivec_tmp[4] = Ty, p_trivec_tmp[5] = Tz;
                    if (use_ray_tube == 1)
                        p_tridir_tmp[2] = az, p_tridir_tmp[3] = el;
                    else
                        p_tridir_tmp[3] = Vx, p_tridir_tmp[4] = Vy, p_tridir_tmp[5] = Vz;
                }
                else if (iTube == 2)
                {
                    p_trivec_tmp[6] = Tx, p_trivec_tmp[7] = Ty, p_trivec_tmp[8] = Tz;
                    if (use_ray_tube == 1)
                        p_tridir_tmp[4] = az, p_tridir_tmp[5] = el;
                    else
                        p_tridir_tmp[6] = Vx, p_tridir_tmp[7] = Vy, p_tridir_tmp[8] = Vz;
                }
            }

            // Calculate the maximum edge length
            if (p_edge_lengthN)
            {
                double Ex = p_trivec_tmp[3] - p_trivec_tmp[0], Ey = p_trivec_tmp[4] - p_trivec_tmp[1], Ez = p_trivec_tmp[5] - p_trivec_tmp[2];
                double scl = Ex * Ex + Ey * Ey + Ez * Ez;
                edge_length_tmp = (scl > edge_length_tmp) ? scl : edge_length_tmp;
                Ex = p_trivec_tmp[6] - p_trivec_tmp[0], Ey = p_trivec_tmp[7] - p_trivec_tmp[1], Ez = p_trivec_tmp[8] - p_trivec_tmp[2];
                scl = Ex * Ex + Ey * Ey + Ez * Ez;
                edge_length_tmp = (scl > edge_length_tmp) ? scl : edge_length_tmp;
                Ex = p_trivec_tmp[6] - p_trivec_tmp[3], Ey = p_trivec_tmp[7] - p_trivec_tmp[4], Ez = p_trivec_tmp[8] - p_trivec_tmp[5];
                scl = Ex * Ex + Ey * Ey + Ez * Ez;
                edge_length_tmp = (scl > edge_length_tmp) ? scl : edge_length_tmp;
                edge_length_tmp = std::sqrt(edge_length_tmp);
            }
        }

        // Re-evaluate coefficients if the ray tube introduced TIR (type-2 tube vertices).
        // A glued sheet keeps its composite: the reflect port stands as it is and the transmit port
        // becomes the leak chain, exactly as for a spine past its critical angle. It can only fire
        // for cases 2/3, where near and far differ; the cases where near == far never reach it.
        if (total_reflection != tir_central)
        {
            if (sheet_resolve)
                SHEET_COMPOSITE(true);
            else
                M1.interact_with(M2, interaction_type, thetaF, fGHz, &cTE, &cTM, nullptr, nullptr, nullptr, nullptr, true, nullptr, true);
        }

        // Read the output ray index
        if (i_rayN >= n_rayN) // Just to be sure to avoid any segfaults
            throw std::invalid_argument("Something went wrong. This should never be reached!");

        // Write ray_indN (inverse of the compaction map, 0-based input ray index)
        if (p_ray_indN)
            p_ray_indN[i_rayN] = (unsigned)iRx;

        // Relaunch offset: a few float ULP at the FBS coordinate magnitude, so the nudge survives the
        // dtype(origN) store and the relaunched ray starts off the face without re-hitting it
        double relaunch_offset = qd_relaunch_offset(Fx, Fy, Fz); // ~8 float ULP at this magnitude
        if (geometry_type != 0 && colocated_faces)           // Relaunch @ SBS
            SET3(p_origN, Sx + relaunch_offset * FDx, Sy + relaunch_offset * FDy, Sz + relaunch_offset * FDz);
        else // Relaunch @ FBS
            SET3(p_origN, Fx + relaunch_offset * FDx, Fy + relaunch_offset * FDy, Fz + relaunch_offset * FDz);

        // Update FD_length to lay beyond the relaunch_offset
        double min_seg = std::max(colocation_dist, 2.0 * relaunch_offset);
        FD_length = (FD_length < min_seg) ? min_seg : FD_length;
        SET3(p_destN, Fx + FD_length * FDx, Fy + FD_length * FDy, Fz + FD_length * FDz);

        SET3(p_fbsN, Fx, Fy, Fz);
        SET3(p_sbsN, Sx, Sy, Sz);

        // Write path_dirN: spine output direction (FD) for reflection/refraction (geometry 0/2, follows a
        // forced-TIR pass-through); spine Snell for undeviated transmission (geometry 1), or undeviated
        // when the spine is in TIR (no Snell direction; any TF leak travels undeviated).
        if (p_path_dirN)
        {
            double PDx = FDx, PDy = FDy, PDz = FDz;
            if (geometry_type == 1 && !tir_central && !no_bend) // spine Snell; under spine TIR there is no Snell direction
                qd_refract(ODx, ODy, ODz, Nx, Ny, Nz, eta, abs_cos_theta, cos_theta2, PDx, PDy, PDz);
            SET3(p_path_dirN, PDx, PDy, PDz);
        }

        if (p_xprmatN || p_gainN)
        {
            double xprmat[8], pgain;
            qd_polbasis(ODx, ODy, ODz, FDx, FDy, FDz, Nx, Ny, Nz, 1.0, cTE, cTM, is_scalar, xprmat, pgain);
            SETL(p_xprmatN, xprmat, nXPR, false);
            SET1(p_gainN, pgain);
        }

        if (use_ray_tube)
            SETL(p_trivecN, p_trivec_tmp);
        if (use_ray_tube == 1) // Spherical
            SETL(p_tridirN, p_tridir_tmp, 6);
        else // Cartesian
            SETL(p_tridirN, p_tridir_tmp);

        SET1(p_fbs_angleN, thetaF);
        SET1(p_thicknessN, FS_length);
        SET1(p_edge_lengthN, edge_length_tmp);

        if (p_normal_vecN)
            SET3(p_normal_vecN, Nx, Ny, Nz), SET3(&p_normal_vecN[3 * n_rayN], Mx, My, Mz);

        if (p_out_typeN)
        {
            out_flags[5] = total_reflection; // Set total reflection flag
            p_out_typeN[i_rayN] = (uint8_t)out_flags;
        }
    }
}

template void quadriga_lib::ray_mesh_interact(int interaction_type, float center_frequency,
                                              const arma::Mat<float> *orig, const arma::Mat<float> *dest,
                                              const arma::Mat<float> *mesh, const arma::uvec *mtl_ind,
                                              const std::unordered_map<std::string, std::vector<float>> *mtl_prop,
                                              const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
                                              const arma::Mat<float> *trivec, const arma::Mat<float> *tridir,
                                              arma::Mat<float> *origN, arma::Mat<float> *destN,
                                              arma::Mat<float> *fbsN, arma::Mat<float> *sbsN,
                                              arma::Col<float> *gainN, arma::Mat<float> *xprmatN,
                                              arma::Mat<float> *trivecN, arma::Mat<float> *tridirN,
                                              arma::Col<float> *fbs_angleN, arma::Col<float> *thicknessN, arma::Col<float> *edge_lengthN,
                                              arma::Mat<float> *normal_vecN, std::vector<uint8_t> *out_typeN,
                                              arma::Mat<float> *path_dirN, bool compact, arma::u32_vec *ray_indN);

template void quadriga_lib::ray_mesh_interact(int interaction_type, double center_frequency,
                                              const arma::Mat<double> *orig, const arma::Mat<double> *dest,
                                              const arma::Mat<double> *mesh, const arma::uvec *mtl_ind,
                                              const std::unordered_map<std::string, std::vector<double>> *mtl_prop,
                                              const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
                                              const arma::Mat<double> *trivec, const arma::Mat<double> *tridir,
                                              arma::Mat<double> *origN, arma::Mat<double> *destN,
                                              arma::Mat<double> *fbsN, arma::Mat<double> *sbsN,
                                              arma::Col<double> *gainN, arma::Mat<double> *xprmatN,
                                              arma::Mat<double> *trivecN, arma::Mat<double> *tridirN,
                                              arma::Col<double> *fbs_angleN, arma::Col<double> *thicknessN, arma::Col<double> *edge_lengthN,
                                              arma::Mat<double> *normal_vecN, std::vector<uint8_t> *out_typeN,
                                              arma::Mat<double> *path_dirN, bool compact, arma::u32_vec *ray_indN);

/*!MD
# ray_state_update
Batched inside/outside ray-state machine with analytic thin-slab (Fabry-Perot) resolution

- Corrects the per-interaction `gainN` / `xprmatN` produced by [[ray_mesh_interact]] using a tracked
  per-ray medium state, and carries that state forward. Three signed-`short` words per ray hold the
  current medium, the previous medium, and a one-slot next-transition buffer. The material index is
  14 bits (`mat = w & 0x3FFF`, at most 16383 materials) and bits 14 and 15 are flags. With bit 15 set
  the buffer word is not a medium: it marks a leg that was entered by crossing a bonded mass-law body
  and its low 15 bits carry a thickness code.
- Implements the inside/outside state machine and overlays a closed-form thin-slab factor `S` (the Airy
  sum) so a single coefficient captures the full internal multiple-reflection series of a parallel slab
  thin enough to matter, instead of relying on the tracer to follow every internal bounce.
- Called twice per interaction by the ray tracer: once for the reflection pass (`interaction_type` 0
  or 3) and once for the transmission/refraction pass (`interaction_type` 1, 2, 4, 5). With `S`
  suppressed (the survival gate re-emits) the transmission/refraction path reproduces [[calc_diffraction_gain]]
- Thin sheets: a `thk > 0` crossing arrives from [[ray_mesh_interact]] already resolved as one atomic
  three-medium event, but with its bounding media guessed from face geometry alone — air on any side the
  mesh did not name. This function supplies the media the tracked state actually reports and re-resolves
  the composite against them, which matters for every sheet that does not sit in air: a sheet embedded in
  a medium is bounded by that medium on both sides, and the wrong pair moves both Fresnel boundaries,
  the internal series and the lumped interface gain at once. The sheet material never becomes the current
  medium; it is recorded in the previous-medium word so the host layer on the far side reflects off it —
  except where the sheet and the host are the same medium electrically, in which case the sheet is not a
  boundary at all, the host layer is carried through it uninterrupted, and its series resolves once over
  the whole path rather than collapsing at a face that does not reflect.
  The total-reflection verdict and the continuation direction are re-derived here too, at the corrected
  angle, for the same reason [[ray_state_update]] re-derives them for a traced solid: an undeviated
  tracer inside a medium travels at its external angle, so the hit code's geometric-angle verdict turns
  a glued exit into a mirror past the internal critical angle, and leaving `path_dirN` unchanged would
  hand the next interaction a geometric direction where a refracted one is required.

## Declaration:
```
void quadriga_lib::ray_state_update(
    int interaction_type,
    dtype center_frequency,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *fbsN,
    const arma::Mat<dtype> *sbsN,
    const arma::u32_vec *no_interact,
    const arma::Col<dtype> *fbs_angleN,
    const arma::Mat<dtype> *normal_vecN,
    const std::vector<uint8_t> *out_typeN,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::Col<short> *mtl_ind_fbsN,
    const arma::Col<short> *mtl_ind_sbsN,
    const arma::Col<short> *mtl_ind_prev_in = nullptr,
    const arma::Col<short> *mtl_ind_current_in = nullptr,
    const arma::Col<short> *mtl_ind_buffer_in = nullptr,
    const arma::Mat<dtype> *path_dir_prev = nullptr,
    const arma::Col<dtype> *acc_dist_geometric = nullptr,
    const arma::Col<dtype> *acc_dist_refracted = nullptr,
    arma::Col<short> *mtl_ind_prev_outN = nullptr,
    arma::Col<short> *mtl_ind_current_outN = nullptr,
    arma::Col<short> *mtl_ind_buffer_outN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,
    arma::Col<dtype> *acc_dist_geometricN = nullptr,
    arma::Col<dtype> *acc_dist_refractedN = nullptr,
    std::vector<uint8_t> *resolved_typeN = nullptr,
    const arma::u32_vec *ray_indN = nullptr,
    double eps = 0.15,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    const arma::Mat<dtype> *tridir = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    const std::vector<uint8_t> *verdict_inN = nullptr,
    std::vector<uint8_t> *verdict_outN = nullptr);
```

## Inputs:
- **`interaction_type`** — 0 EM reflection, 1 EM transmission, 2 EM refraction, 3 scalar reflection, 4 scalar transmission, 5 scalar refraction
- **`center_frequency`** — Center frequency in [Hz]
- **`orig`**, **`dest`** — Ray origin, destination, full ray set; `[n_ray, 3]`, read at `g = ray_indN[i]`
- **`fbsN`**, **`sbsN`**  — First and second interaction points, compact set; `[n_rayN, 3]`
- **`no_interact`** — Mesh-hit count per ray, full ray set; `[n_ray]`
- **`fbs_angleN`** — Incidence angle at FBS (ITU convention), compact set; `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normals `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`, compact set; `[n_rayN, 6]`.
  The VBS plane normal for the Snell corrections; currently also gates the parallelism (wedge) test.
  NULL disables the wedge test.
- **`out_typeN`** — Interaction type code from [[ray_mesh_interact]], compact set; `[n_rayN]`. Bit 5 is
  stripped and re-evaluated at the VBS angle; bit 6 (thin sheet) is dispatched on and is *not* stripped,
  so the sheet codes 65 / 67 / 69 / 71 / 79 take their own path ahead of the solid state machine.
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]).
  The `thk` column is read here as well as in [[ray_mesh_interact]]; it selects the thin-sheet path.
- **`mtl_ind_fbsN`**, **`mtl_ind_sbsN`** — Material indices M1 / M2 of the FBS / SBS faces, compact set; `[n_rayN]` (0 = air).
  `mtl_ind_sbsN` is load-bearing beyond media-to-media: it carries the second face of a glued or duplicate
  sheet pair, and the thin-sheet case is re-derived from
  `(code, is_sheet(FBS), is_sheet(SBS), same material)` because codes 69 and 71 are shared by three topologies.
- **`mtl_ind_prev_in`**, **`mtl_ind_current_in`**, **`mtl_ind_buffer_in`** — State words, full ray set; `[n_ray]`,
  read at `g`, never written. NULL reads as state `0` (outside, no flags). A buffer word with bit 15 set is
  a surface-mass code written by an earlier call and is exempt from the material range check; pass the
  words through unchanged between calls. A leg that carries the followed mark forms no series: the
  verdict of a body is taken once, at the first event that closes a leg in it.<br><br>
   | Word                 | Bits 0-13                          | Bit 14                                        | Bit 15                    |
   | -------------------- | ---------------------------------- | --------------------------------------------- | ------------------------- |
   | `mtl_ind_current`    | medium the ray is in, 0 = air      | bent: the leg follows a bend traced with type 2 or 5; zero in air | resolved |
   | `mtl_ind_prev`       | medium the leg was entered from, or the sheet crossed or reflected off | the sheet in this word lies inside the current medium | known non-parallel |
   | `mtl_ind_buffer`     | deferred medium                    | followed, with a zero index: the leg began at a reflection that left the series to the tracer | clear |
   | `mtl_ind_buffer`     | thickness code, bits 0-14          | (part of the code)                            | set: mass-crossed         |
   Under bit 15 the code is the normal thickness of a mass-law leg closed on the way into the current
   one: of the solid in `mtl_ind_prev`, or, where that word holds a sheet lying inside the current
   medium, of the part of the current medium in front of that sheet. Zero means no such leg.
- **`path_dir_prev`** — Physical ray direction entering this segment, full ray set; `[n_ray, 3]`
- **`acc_dist_geometric`** — Accumulated geometric in-layer distance carried into this call, full ray set; `[n_ray]`.
  NULL reads as 0. Frequency-independent: built from the traced `orig`-`fbs` / `fbs`-`dest` lengths only, so a
  multi-frequency caller keeps a single shared instance.
- **`acc_dist_refracted`** — Accumulated refracted (VBS) in-layer distance carried into this call, full ray set; `[n_ray]`.
  NULL reads as 0. Frequency-dependent, since the Snell bend that defines it follows the material at
  `center_frequency`, so a multi-frequency caller keeps one instance per frequency. The two inputs are
  independent; either may be NULL.
- **`ray_indN`** — Compact-to-full ray index map; `[n_rayN]` to `[n_ray]`; NULL = identity (`n_ray == n_rayN`)
- **`eps`** *(optional)* — Thin-slab (Fabry-Pérot) resolve threshold on the round-trip in-slab
  amplitude `ρ` (`ρ ∈ [0, 1]`): the closed-form Airy factor is applied when `ρ ≥ eps` and the series
  is re-emitted to the tracer when `ρ < eps` (weak / fast-decaying slabs). `eps = 0` always resolves
  (for callers that cannot re-emit, e.g. [[calc_diffraction_gain]]); `eps ≥ 1` always re-emits
  (resolution disabled). A near-pole `S` or a known non-parallel slab re-emits regardless of `eps`.
  Does **not** gate a thin sheet's own series: a sheet has no interior for the tracer to re-emit into, so
  that series always resolves. `eps` still gates the host layer a sheet is embedded in.
  Default `0.15`.
- **`verdict_inN`** *(optional)* — The verdict of each event on its bounce series, as an earlier call on the
  same events wrote it to `verdict_outN`, compact set; `[n_rayN]`. Where given, a series is summed exactly
  where the entry is nonzero and `eps` is not read. The traced paths of a multi-frequency run are the base
  frequency's - a reflected ray is resolved or followed as that frequency decided - so the caller hands
  the base frequency's verdict to the call of every other frequency. A summed series near a pole is
  clamped in magnitude instead of being re-emitted. NULL: this call decides, by `eps`.
- **`tridir`** *(optional)* — Vertex-ray directions of the beam entering this segment, Cartesian, as handed to
  [[ray_mesh_interact]]; `[n_ray, 9]`. Given together with `tridirN`; requires `origN` and `destN`

## Outputs:
- **`mtl_ind_prev_outN`**, **`mtl_ind_current_outN`**, **`mtl_ind_buffer_outN`** — Updated state words,
  compact set; `[n_rayN]`. `mtl_ind_prev_outN` may hold a **sheet** material index after a thin-sheet
  crossing. That is a boundary material, not a medium the ray travelled through: the previous-medium word
  is only ever read as the far material of a slab resolve, for the parallelism flag, and as the bonded
  mass-law body a leg was entered through, and a sheet is the correct far material for the layer on
  either side of it. `mtl_ind_buffer_outN` with bit 15 set carries the surface-mass code of that body
  (a logarithmic thickness code for a traced solid; for a sheet zero, or the thickness of the host in
  front of a sheet that lies inside it) instead of a deferred medium. Bit 14 of `mtl_ind_prev_outN` is
  set where the sheet in that word lies inside the current medium, so that the layer behind it is
  backed by that medium and not by air.
  NULL skips the write. Passing all six state args NULL disables tracking —
  each interaction is corrected on its own (entry loss, TR kill, single-hit air-gap `S`); cross-interaction slab `S` and
  reflection-bounce `S` need the tracked medium.
- **`gainN`** *(in/out)* — Per-interaction linear power gain, updated in place; `[n_rayN]`. Kept consistent
  with `xprmatN` at every write. A killed ray (`resolved_typeN == 0`) yields `gainN = 0` together with
  an all-zero `xprmatN`.
- **`xprmatN`** *(in/out)* — Polarization transfer matrix, updated in place; `[8, n_rayN]` for EM mode,
  `[2, n_rayN]` for scalar mode. Same layout and basis convention as in [[ray_mesh_interact]], but with the
  medium closed out: on return it additionally contains the in-medium attenuation and excess phase of the
  traversed segment and, when the resolve bit of `resolved_typeN` is set, the closed-form thin-slab (Airy)
  factor `S` covering the full internal multiple-reflection series. `S` is applied per polarization and
  is a pure series factor — the interface coefficients it multiplies are the ones [[ray_mesh_interact]]
  already wrote, and no magnitude conversion is applied on top of them. FSPL / spreading loss remains excluded,
  so the matrix is the complete per-interaction Jones factor and can be left-multiplied directly into a
  per-path product. When the series is re-emitted instead (`ρ < eps`, resolve bit clear), the matrix carries
  only the first-pass coefficient and the remaining bounces arrive as separate interactions.
  On a thin-sheet crossing the matrix is *replaced* by the composite re-resolved for the bounding media the
  state words report, and no medium gain is charged for the sheet material itself — its traversal loss and
  phase are already inside the composite. The host medium the ray was travelling in is still closed out.
- **`path_dirN`** *(in/out)* — Continuation direction, corrected in place by the VBS construction, compact set; `[n_rayN, 3]`
- **`origN`**, **`destN`** *(optional, in/out)* — The relaunch segment [[ray_mesh_interact]] built, compact set;
  `[n_rayN, 3]`, given together. With a refracting type (2, 5) that segment was bent at a real angle in the
  medium the ray leaves. Where this function writes the direction of a wave out of a lossy medium whose
  in-medium cosine it recovered - a bare exit, a glued exit, a transition into another solid - both points
  are put on one line along `path_dirN`, through the SBS of a co-located pair and the FBS otherwise. Every
  other ray, and every other type, is left untouched; NULL leaves the geometry to the caller as before.
  A caller that runs several frequencies passes them on the call that defines the geometry only.
- **`tridirN`** *(optional, in/out)* — Vertex-ray directions leaving, Cartesian, compact set; `[n_rayN, 9]`.
  Corrected together with `origN` and `destN`: each vertex ray gets the construction of the spine from its
  own direction in `tridir`. If one of the three cannot be formed, the ray keeps the geometry it had.
- **`verdict_outN`** *(optional)* — Per event, 1 where this call summed a bounce series and 0 where it left the
  bounces to the tracer, compact set; `[n_rayN]`. NULL skips the write.
- **`acc_dist_geometricN`** — Accumulated geometric distance leaving this call, compact set; `[n_rayN]`. NULL skips the write.
- **`acc_dist_refractedN`** — Accumulated refracted (VBS) distance leaving this call, compact set; `[n_rayN]`. NULL skips the write.
- **`resolved_typeN`** *(optional)* — Resolved interaction-type code, bit-encoded (`qd::bits<uint8_t>`),
  compact set; `[n_rayN]`. 0 = ray killed. NULL skips the write.<br><br>
   | Bit  | Flag        | Meaning                                                                        |
   | :--: | :---------: | ------------------------------------------------------------------------------ |
   |   0  | ok          | OK flag (0 = a deferred degenerate-resolve buffer is pending)                  |
   |   1  | vbs         | VBS correction (gain/xprmat corrected at the VBS instead of FBS/SBS)           |
   |   2  | resolve     | Internal multi-bounce series accounted for (expect no further internal bounces) |
   |   3  | inside      | Inside-object flag (1 = ray continues inside, 0 = continues outside)           |
   |   4  | fix         | Fix flag (resolved-false-outside, or entry/exit material mismatch)             |
   |   5  | tir         | Total-reflection flag (also set when a transmission factor forced reflection)  |
   |   6  | trans/sheet | See the qualification below                                                    |
   |   7  | refl        | Reflection flag (0 = transmission/refraction, 1 = reflection/scattering)       |
   Bit 6 is qualified by bit 2, because a transparent interface resolves no series and scattering is not
   implemented, so the two bits together are unreachable for any non-sheet interaction:<br><br>
   | Bit 6 | Bit 2 | Meaning                                                                       |
   | :---: | :---: | ----------------------------------------------------------------------------- |
   |   1   |   0   | Transparent interface (transmission) / scatter (reflection) — as before       |
   |   1   |   1   | **Thin-sheet crossing**                                                       |
   Inside the thin-sheet region bit 1 does not mean VBS: the crossing is always re-resolved at the VBS
   angle, so a VBS flag would be constant there. It carries the glued verdict instead:<br><br>
   | Bit 1 (sheet region) | Meaning                                                                |
   | :------------------: | ---------------------------------------------------------------------- |
   |          0           | Isolated — the same medium bounds both sides of the sheet              |
   |          1           | Glued — the two bounding media differ                                  |
   The sheet region reports the resolved *state*, not the sheet topology; the state words and `out_typeN`
   separate an embedded sheet from a glued entry. Note that 69 and 71 also occur as `out_typeN` values with
   entirely different meanings — the two fields have always had separate tables.
   Reachable composite values for transmission / refraction:<br><br>
   |  Dec | Hex  | FIX  |   TIR   | Flags set                  | Meaning                                            |
   | :--: | :--: | :--: | :-----: | -------------------------- | -------------------------------------------------- |
   |    9 | 0x09 |   —  |    41   | inside, ok                 | o-i entry, OR i-i transition (refr. 2/5, FBS==VBS) |
   |    8 | 0x08 |   —  |    40   | inside                     | o-i entry, deferred buffer set (overlap/edge)      |
   |   11 | 0x0B |  27  |  43, 59 | inside, vbs, ok            | i-i transition (undev. 1/4, VBS relocated)         |
   |   13 | 0x0D |  29  |  45, 61 | inside, resolve, ok        | i-i transition + slab series (refr. 2/5, FBS==VBS) |
   |   15 | 0x0F |  31  |  47, 63 | inside, resolve, vbs, ok   | i-i transition + slab series (undev. 1/4, VBS)     |
   |    1 | 0x01 |  17  |  33, 49 | ok                         | i-o exit (refr. 2/5, FBS==VBS)                     |
   |    3 | 0x03 |  19  |  35, 51 | vbs, ok                    | i-o exit (undev. 1/4, VBS relocated)               |
   |    5 | 0x05 |  21  |  37, 53 | resolve, ok                | i-o exit + slab series (refr. 2/5, FBS==VBS)       |
   |    7 | 0x07 |  23  |  39, 55 | resolve, vbs, ok           | i-o exit + slab series (undev. 1/4, VBS)           |
   |   73 | 0x49 |  89  |     —   | trans, inside, ok          | ignore-hit / same-medium pass, no gain change      |
   |   72 | 0x48 |  88  |     —   | trans, inside              | nested pass-through, buffer deferred               |
   |   65 | 0x41 |   —  |     —   | trans, ok                  | advance to ray destination, identity interface     |
   |   69 | 0x45 |   85 |   101   | sheet, resolve, ok         | isolated sheet crossed in air                      |
   |   77 | 0x4D |   93 |   109   | sheet, resolve, inside, ok | isolated sheet crossed inside a medium (embedded)  |
   |   71 | 0x47 |   87 |   103   | sheet, glued, resolve, ok  | glued exit, solid \| sheet \| air                  |
   |   79 | 0x4F |   95 |   111   | sheet, glued, resolve, inside, ok | glued entry, host \| sheet \| solid         |
   Reachable composite values for reflection:<br><br>
   |  Dec | Hex  |  FIX |   TIR   | Flags set                       | Meaning                                                      |
   | :--: | :--: | :--: | :-----: | ------------------------------- |------------------------------------------------------------- |
   |  129 | 0x81 |    — |   161   | refl, ok                        | eager front reflection (R0), outside (FBS==VBS)              |
   |  137 | 0x89 |  153 | 169,185 | refl, inside, ok                | internal back-reflection (incoming refr. 2/5, FBS==VBS)      |
   |  139 | 0x8B |  155 | 171,187 | refl, inside, vbs, ok           | internal back-reflection (incoming undev. 1/4, VBS)          |
   |  141 | 0x8D |  157 | 173,189 | refl, inside, resolve, ok       | internal back-reflection + slab series (incoming refr. 2/5)  |
   |  143 | 0x8F |  159 | 175,191 | refl, inside, resolve, vbs, ok  | internal back-reflection + slab series (incoming undev. 1/4) |
   |  197 | 0xC5 |  213 |   229   | refl, sheet, resolve, ok        | reflection off an isolated sheet, ray outside                |
   |  205 | 0xCD |  221 |   237   | refl, sheet, resolve, inside, ok | reflection off an isolated sheet, ray inside a medium       |
   |  199 | 0xC7 |  215 |   231   | refl, sheet, glued, resolve, ok | reflection off a glued sheet from the host side              |
   |  207 | 0xCF |  223 |   239   | refl, sheet, glued, resolve, inside, ok | reflection off a glued sheet from the solid side    |
   |  192+| 0xC0+|    — |    —    | refl, scatter, resolve clear    | reserved: scattering not implemented                         |
   A thin-sheet stack of two different materials (`out_typeN` 79, or 69/71 with two sheet faces of
   different materials) is **not implemented** and returns 0: the case is detected but has no physics in
   [[ray_mesh_interact]] either.


## See also:
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-State Machine</a> (companion document)
- [[ray_mesh_interact]] (computes the per-interaction Fresnel/Jones result this function corrects)
- [[calc_diffraction_gain]] (the reference state machine this function ports)
MD!*/

template <typename dtype>
void quadriga_lib::ray_state_update(int interaction_type,
                                    dtype center_frequency,
                                    const arma::Mat<dtype> *orig,
                                    const arma::Mat<dtype> *dest,
                                    const arma::Mat<dtype> *fbsN,
                                    const arma::Mat<dtype> *sbsN,
                                    const arma::u32_vec *no_interact,
                                    const arma::Col<dtype> *fbs_angleN,
                                    const arma::Mat<dtype> *normal_vecN,
                                    const std::vector<uint8_t> *out_typeN,
                                    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
                                    const arma::Col<short> *mtl_ind_fbsN,
                                    const arma::Col<short> *mtl_ind_sbsN,
                                    const arma::Col<short> *mtl_ind_prev_in,
                                    const arma::Col<short> *mtl_ind_current_in,
                                    const arma::Col<short> *mtl_ind_buffer_in,
                                    const arma::Mat<dtype> *path_dir_prev,
                                    const arma::Col<dtype> *acc_dist_geometric,
                                    const arma::Col<dtype> *acc_dist_refracted,
                                    arma::Col<short> *mtl_ind_prev_outN,
                                    arma::Col<short> *mtl_ind_current_outN,
                                    arma::Col<short> *mtl_ind_buffer_outN,
                                    arma::Col<dtype> *gainN,
                                    arma::Mat<dtype> *xprmatN,
                                    arma::Mat<dtype> *path_dirN,
                                    arma::Col<dtype> *acc_dist_geometricN,
                                    arma::Col<dtype> *acc_dist_refractedN,
                                    std::vector<uint8_t> *resolved_typeN,
                                    const arma::u32_vec *ray_indN,
                                    double eps,
                                    arma::Mat<dtype> *origN,
                                    arma::Mat<dtype> *destN,
                                    const arma::Mat<dtype> *tridir,
                                    arma::Mat<dtype> *tridirN,
                                    const std::vector<uint8_t> *verdict_inN,
                                    std::vector<uint8_t> *verdict_outN)
{
    if (interaction_type < 0 || interaction_type > 5)
        throw std::invalid_argument("Interaction type must be either (0) EM Reflection, (1) EM Transmission, (2) EM Refraction, (3) Scalar Reflection, (4) Scalar Transmission, (5) Scalar Refraction");
    const bool is_scalar = interaction_type >= 3;
    const bool refl_pass = (interaction_type == 0 || interaction_type == 3); // geometry 0
    const arma::uword nXPR = is_scalar ? 2 : 8;                              // NUmber of columns in xprmat (8 for EM, 2 for scalar)

    if (!std::isfinite((double)center_frequency) || center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");

    if (!std::isfinite(eps) || eps < 0.0)
        throw std::invalid_argument("Input 'eps' must be finite and >= 0.");

    if (orig == nullptr || dest == nullptr || fbsN == nullptr || sbsN == nullptr)
        throw std::invalid_argument("Inputs 'orig', 'dest', 'fbsN' and 'sbsN' cannot be NULL.");
    if (orig->n_cols != 3 || dest->n_cols != 3 || fbsN->n_cols != 3 || sbsN->n_cols != 3)
        throw std::invalid_argument("Inputs 'orig', 'dest', 'fbsN' and 'sbsN' must have 3 columns.");
    const arma::uword n_ray = orig->n_rows;
    const arma::uword n_rayN = fbsN->n_rows;

    if (dest->n_rows != n_ray)
        throw std::invalid_argument("Inputs 'orig' and 'dest' must have the same number of rows.");
    if (sbsN->n_rows != n_rayN)
        throw std::invalid_argument("Inputs 'fbs' and 'sbs' must have the same number of rows.");

    if (no_interact == nullptr || no_interact->n_elem != n_ray)
        throw std::invalid_argument("Input 'no_interact' must match the number of rays in 'orig'.");
    const unsigned *p_no_interact = no_interact->memptr();

    if (fbs_angleN == nullptr || fbs_angleN->n_elem != n_rayN)
        throw std::invalid_argument("Input 'fbs_angleN' cannot be NULL and have length n_rayN.");
    const dtype *p_fbs_angleN = fbs_angleN->memptr();

    if (normal_vecN == nullptr)
        throw std::invalid_argument("Input 'normal_vecN' is required (VBS plane normal).");
    if (normal_vecN->n_rows != n_rayN || normal_vecN->n_cols != 6)
        throw std::invalid_argument("Input 'normal_vecN' must have size [n_rayN, 6].");
    const dtype *p_normal_vecN = normal_vecN->memptr();

    if (out_typeN == nullptr || out_typeN->size() != n_rayN)
        throw std::invalid_argument("Input 'out_typeN' cannot be NULL and have length n_rayN.");
    const uint8_t *p_out_typeN = out_typeN->data();

    // Resolved material columns for the Material(cols, idx)
    MaterialCols<dtype> cols = mtl_prop ? MaterialCols<dtype>(*mtl_prop) : MaterialCols<dtype>();
    if (cols.n_mtl > 16383) // the state words carry a 14-bit material index
        throw std::invalid_argument("Number of materials cannot exceed 16383.");

    if (mtl_ind_fbsN == nullptr || mtl_ind_fbsN->n_elem != n_rayN)
        throw std::invalid_argument("Input 'mtl_ind_fbs' must match the length of 'out_typeN'.");
    const short *p_mtl_ind_fbs = mtl_ind_fbsN->memptr();
    if (mtl_ind_fbsN->n_elem != 0 && arma::uword(mtl_ind_fbsN->max() & (short)0x7FFF) > cols.n_mtl)
        throw std::invalid_argument("Values in 'mtl_ind_fbs' exceed the number of materials in 'mtl_prop'.");

    if (mtl_ind_sbsN == nullptr || mtl_ind_sbsN->n_elem != n_rayN)
        throw std::invalid_argument("Input 'mtl_ind_sbs' must match the length of 'out_typeN'.");
    const short *p_mtl_ind_sbs = mtl_ind_sbsN->memptr();
    if (mtl_ind_sbsN->n_elem != 0 && arma::uword(mtl_ind_sbsN->max() & (short)0x7FFF) > cols.n_mtl)
        throw std::invalid_argument("Values in 'mtl_ind_sbs' exceed the number of materials in 'mtl_prop'.");

    // The buffer word under its flag bit carries a surface-mass code, not a material index.
    // Without it, bit 14 is the followed mark, as bit 14 of the other two words is a flag.
    auto check_state_words = [&](const arma::Col<short> *v, const char *name, bool flag_is_code = false)
    {
        if (v == nullptr || v->n_elem == 0)
            return;
        for (const short *p = v->memptr(), *pe = p + v->n_elem; p < pe; ++p)
            if (!(flag_is_code && (*p & (short)0x8000)) && arma::uword(*p & (short)0x3FFF) > cols.n_mtl)
                throw std::invalid_argument(std::string("Values in '") + name + "' exceed the number of materials in 'mtl_prop'.");
    };
    check_state_words(mtl_ind_prev_in, "mtl_ind_prev_in");
    check_state_words(mtl_ind_current_in, "mtl_ind_current_in");
    check_state_words(mtl_ind_buffer_in, "mtl_ind_buffer_in", true);

    if (mtl_ind_prev_in && mtl_ind_prev_in->n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_prev_in' must match the number of rays in 'orig'.");
    const short *p_prev_in = mtl_ind_prev_in ? mtl_ind_prev_in->memptr() : nullptr;

    if (mtl_ind_current_in && mtl_ind_current_in->n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_current_in' must match the number of rays in 'orig'.");
    const short *p_cur_in = mtl_ind_current_in ? mtl_ind_current_in->memptr() : nullptr;

    if (mtl_ind_buffer_in && mtl_ind_buffer_in->n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_buffer_in' must match the number of rays in 'orig'.");
    const short *p_buf_in = mtl_ind_buffer_in ? mtl_ind_buffer_in->memptr() : nullptr;

    if (path_dir_prev && (path_dir_prev->n_rows != n_ray || path_dir_prev->n_cols != 3))
        throw std::invalid_argument("Input 'path_dir_prev' must have size [n_ray, 3].");
    const dtype *p_path_dir_prev = path_dir_prev ? path_dir_prev->memptr() : nullptr;

    if (acc_dist_geometric && acc_dist_geometric->n_elem != n_ray)
        throw std::invalid_argument("Input 'acc_dist_geometric' must match the number of rays in 'orig'.");
    const dtype *p_acc_geo_in = acc_dist_geometric ? acc_dist_geometric->memptr() : nullptr;

    if (acc_dist_refracted && acc_dist_refracted->n_elem != n_ray)
        throw std::invalid_argument("Input 'acc_dist_refracted' must match the number of rays in 'orig'.");
    const dtype *p_acc_refract_in = acc_dist_refracted ? acc_dist_refracted->memptr() : nullptr;

    if (mtl_ind_prev_outN && mtl_ind_prev_outN->n_elem != n_rayN)
        mtl_ind_prev_outN->set_size(n_rayN);
    short *p_prev_outN = mtl_ind_prev_outN ? mtl_ind_prev_outN->memptr() : nullptr;

    if (mtl_ind_current_outN && mtl_ind_current_outN->n_elem != n_rayN)
        mtl_ind_current_outN->set_size(n_rayN);
    short *p_cur_outN = mtl_ind_current_outN ? mtl_ind_current_outN->memptr() : nullptr;

    if (mtl_ind_buffer_outN && mtl_ind_buffer_outN->n_elem != n_rayN)
        mtl_ind_buffer_outN->set_size(n_rayN);
    short *p_buf_outN = mtl_ind_buffer_outN ? mtl_ind_buffer_outN->memptr() : nullptr;

    if (gainN && gainN->n_elem != n_rayN)
        throw std::invalid_argument("In-out 'gainN' must match the length of 'out_typeN'.");
    dtype *p_gainN = gainN ? gainN->memptr() : nullptr;

    if (xprmatN && (xprmatN->n_rows != nXPR || xprmatN->n_cols != n_rayN))
        throw std::invalid_argument("In-out 'xprmatN' must have size [8, n_rayN] for EM mode or [2, n_rayN] for scalar mode.");
    dtype *p_xprmatN = xprmatN ? xprmatN->memptr() : nullptr;

    if (path_dirN && (path_dirN->n_rows != n_rayN || path_dirN->n_cols != 3))
        throw std::invalid_argument("In-out 'path_dirN' must have size [n_rayN, 3].");
    dtype *p_path_dirN = path_dirN ? path_dirN->memptr() : nullptr;

    if (acc_dist_geometricN && acc_dist_geometricN->n_elem != n_rayN)
        acc_dist_geometricN->set_size(n_rayN);
    dtype *p_acc_geo_outN = acc_dist_geometricN ? acc_dist_geometricN->memptr() : nullptr;

    if (acc_dist_refractedN && acc_dist_refractedN->n_elem != n_rayN)
        acc_dist_refractedN->set_size(n_rayN);
    dtype *p_acc_refract_outN = acc_dist_refractedN ? acc_dist_refractedN->memptr() : nullptr;

    if (resolved_typeN && resolved_typeN->size() != n_rayN)
        resolved_typeN->resize(n_rayN, 0); // Init to 0
    uint8_t *p_resolved_typeN = resolved_typeN ? resolved_typeN->data() : nullptr;

    if (ray_indN && ray_indN->n_elem != n_rayN)
        throw std::invalid_argument("Input 'ray_indN' must match the length of 'out_typeN'.");
    if (ray_indN && ray_indN->n_elem != 0 && (arma::uword)ray_indN->max() >= n_ray)
        throw std::invalid_argument("Values in 'ray_indN' exceed the number of rays in 'orig'.");
    if (ray_indN == nullptr && n_ray != n_rayN)
        throw std::invalid_argument("Without 'ray_indN', the full and compact sets must have the same size.");
    const unsigned *p_ray_indN = ray_indN ? ray_indN->memptr() : nullptr;

    // The relaunch geometry ray_mesh_interact built, corrected in place where a direction is
    // written out of a lossy medium with a refracting type. Optional: without it nothing moves.
    if ((origN == nullptr) != (destN == nullptr))
        throw std::invalid_argument("In-out 'origN' and 'destN' must be provided together.");
    if (origN && (origN->n_rows != n_rayN || origN->n_cols != 3 || destN->n_rows != n_rayN || destN->n_cols != 3))
        throw std::invalid_argument("In-out 'origN' and 'destN' must have size [n_rayN, 3].");
    if ((tridir == nullptr) != (tridirN == nullptr))
        throw std::invalid_argument("Input 'tridir' and in-out 'tridirN' must be provided together.");
    if (tridir && origN == nullptr)
        throw std::invalid_argument("Input 'tridir' requires 'origN' and 'destN'.");
    if (tridir && (tridir->n_rows != n_ray || tridir->n_cols != 9))
        throw std::invalid_argument("Input 'tridir' must have size [n_ray, 9] (Cartesian).");
    if (tridirN && (tridirN->n_rows != n_rayN || tridirN->n_cols != 9))
        throw std::invalid_argument("In-out 'tridirN' must have size [n_rayN, 9] (Cartesian).");
    dtype *p_origN = origN ? origN->memptr() : nullptr;
    dtype *p_destN = destN ? destN->memptr() : nullptr;
    const dtype *p_tridir = tridir ? tridir->memptr() : nullptr;
    dtype *p_tridirN = tridirN ? tridirN->memptr() : nullptr;

    // The verdict of each event on its bounce series, handed from the base frequency of a run to
    // the others. Optional: without it every call decides for itself.
    if (verdict_inN && verdict_inN->size() != n_rayN)
        throw std::invalid_argument("Input 'verdict_inN' must match the length of 'out_typeN'.");
    const uint8_t *p_verdict_inN = verdict_inN ? verdict_inN->data() : nullptr;
    if (verdict_outN && verdict_outN->size() != n_rayN)
        verdict_outN->resize(n_rayN, 0);
    uint8_t *p_verdict_outN = verdict_outN ? verdict_outN->data() : nullptr;

#pragma omp parallel for schedule(static)
    for (long long i_rayN = 0; i_rayN < (long long)n_rayN; ++i_rayN) // Interaction loop (compact set)
    {
        size_t ii = (size_t)i_rayN;
        size_t i_ray = p_ray_indN ? (size_t)p_ray_indN[ii] : ii; // Full-set index

        // Resolved interaction type classifier
        // Default to: transmission, outside, ok (no resolve, no vbs, not transparent)
        qd::bits<uint8_t> typeR = 1; // Set OK, no-buffer

        // Initialize output state words from current ones (will be overwritten)
        short next_current = p_cur_in ? p_cur_in[i_ray] : (short)0;
        short next_prev = p_prev_in ? p_prev_in[i_ray] : (short)0;
        short next_buffer = p_buf_in ? p_buf_in[i_ray] : (short)0;

        // The three state words, decoded here and nowhere else. The material index is 14 bits.
        //
        //   word      bits 0-13                      bit 14            bit 15
        //   current   medium the ray is in           bent              resolved
        //   prev      medium the leg was entered     prev_embedded     prev_nonpar
        //             from, or the sheet crossed     (a sheet), or
        //             or reflected off               prev_real_inv
        //                                            (a solid)
        //   buffer    deferred medium                followed          clear
        //   buffer    thickness code, bits 0-14                        set: mass_crossed
        //
        // prev_embedded qualifies a sheet in the previous-medium word: the sheet lies inside the
        // current medium, so what is behind it is that medium. Clear, the sheet is a lining and
        // what is behind it is air. It is written by a sheet event whose two sides are one medium
        // and travels with the word.
        //
        // The same bit qualifies a solid in that word, as prev_real_inv: the shared face was
        // evaluated at the recovered cosine of a wave whose transverse invariant is real, so the
        // wave in the current medium has a real invariant as well and the recovery solves for it
        // without naming the solid. Clear, the face was evaluated at a real angle in the solid and
        // the recovery inverts that. It is written by a transition out of a lossy solid and
        // travels with the word; which of the two the bit means follows from what the word holds.
        //
        // bent says that the leg the ray is on follows a bend the tracer traced: the medium was
        // entered by a refracting event (type 2 or 5) that was not a total reflection, so the
        // segment is the refracted direction itself and the traced and the geometric cosine
        // coincide. It is a statement about the polyline, which is bent at the base frequency
        // and shared by every frequency of a run, as the word is. Written by the event that
        // enters the medium, carried while the ray stays in it - a reflection, a sheet inside
        // the medium, a transparent pass - and zero in air. The undeviated tracer never sets it.
        int mtl_current = next_current & 0x3FFF;
        int mtl_prev = next_prev & 0x3FFF;
        int mtl_buffer = next_buffer & 0x7FFF;
        bool resolved = (next_current & 0x8000) != 0;
        const bool bent = (next_current & 0x4000) != 0 && mtl_current != 0;
        const int bent_carry = bent ? (int)0x4000 : 0;
        bool prev_nonpar = (next_prev & 0x8000) != 0;
        const bool prev_embedded = (next_prev & 0x4000) != 0;

        // The buffer word under its flag bit is not a deferred medium. It then says that the leg
        // the ray is on was entered by crossing a bonded mass-law body, the one in the
        // previous-medium word, and its low 15 bits carry the surface mass the material table
        // does not: a logarithmic code of the normal thickness of a leg that was closed,
        // d = 10^(-4 + 6 (k - 1) / 32766) m for k = 1 .. 32767, which is 0.1 mm to 100 m in steps
        // of 0.044 %. For a traced solid in the previous-medium word it is that solid's leg. A
        // sheet's own thickness is a column, so for a sheet the code is zero, or, where the sheet
        // lies inside the current medium, the leg of that medium in front of the sheet, which the
        // leg behind it is bonded to as well. Bonded bodies crossed before that leg which have its
        // slope m are in the thickness too, as what they are worth in its material; see MASS_FOLD.
        // The thickness is geometric, so one code serves every frequency. It is written where a mass-law body is crossed into another medium, kept
        // where a leg carries on, and cleared wherever a leg begins any other way, a reflection
        // included, so a set flag always means the neighbour was crossed on this traversal. Every
        // reader of the deferred medium sees zero while the flag is set.
        const short buffer_word_in = next_buffer;
        const bool mass_crossed = (next_buffer & 0x8000) != 0;
        const int mass_code = mass_crossed ? mtl_buffer : 0;

        // followed: the leg the ray is on began at a reflection inside this medium that left the
        // bounce series to the tracer. Whether a body's series is summed or followed is one
        // verdict, taken by the first event that closes a leg in it, where the previous-medium
        // word names the mirror behind the ray. On a later leg that mirror is the face the ray was
        // reflected off, which no word holds, and the pair is the one that was already judged:
        // the leg forms no series. Written by a reflection that did not sum, in the buffer word,
        // which every reflection clears; carried where a leg carries on; replaced by a deferred
        // medium that has to be stored, after which the leg judges for itself again.
        const bool followed = !mass_crossed && (next_buffer & 0x4000) != 0;
        if (mass_crossed || followed)
            mtl_buffer = 0;

        // Compact-set reads at i
        unsigned nH = p_no_interact[i_ray];                // No interactions
        auto typeH = (qd::bits<uint8_t>)p_out_typeN[ii];   // Hit type code from ray_mesh_interact
        int M1 = (int)(p_mtl_ind_fbs[ii] & (short)0x3FFF); // FBS material
        int M2 = (int)(p_mtl_ind_sbs[ii] & (short)0x3FFF); // SBS material
        double theta = (double)p_fbs_angleN[ii];           // Incidence angle @ FBS
        double fGHz = (double)center_frequency * 1e-9;     // Frequency in GHz

        // Read total reflection condition and strip TR flag from typeH
        bool total_reflection = typeH[5]; // Read from hit type
        typeH[5] = false;                 // Clear from hit type
        typeR[5] = total_reflection;      // Set on resolved type

        // Build a Material for a 1-based index (0 = air). Used by SAME, TRN and the slab-factor calls.
        auto MAT = [&](int m) -> Material
        { return Material(cols, (arma::uword)(m < 1 ? 0 : m)); };

        // Same-medium test. Index 0 is the "outside / no medium" sentinel, not a material, so it
        // only matches itself; two real materials match on identical properties.
        auto SAME = [&](int a, int b) -> bool
        { return a == b || (a > 0 && b > 0 && MAT(a).same_as(MAT(b))); };

        // Electrical indistinguishability at this frequency. Two materials that agree in permittivity
        // and permeability present no boundary to a wave, whatever their other columns say.
        // Material::same_as cannot be used for this: it compares thk, which is exactly the column
        // that differs between a sheet and the solid it is cut from, and thk has no effect on a
        // boundary. Used to decide whether a sheet crossing actually interrupts the host layer.
        auto TRANSPARENT_TO = [&](int a, int b) -> bool
        {
            Material A = MAT(a), B = MAT(b);
            std::complex<double> ea = A.eta(fGHz) + A.eta_resonance(fGHz);
            std::complex<double> eb = B.eta(fGHz) + B.eta_resonance(fGHz);
            std::complex<double> ma = A.mu(fGHz), mb = B.mu(fGHz);
            double se = std::abs(ea) + std::abs(eb), sm = std::abs(ma) + std::abs(mb);
            return std::abs(ea - eb) <= 1.0e-9 * ((se > 0.0) ? se : 1.0) &&
                   std::abs(ma - mb) <= 1.0e-9 * ((sm > 0.0) ? sm : 1.0);
        };

        // Thin-sheet test for a 1-based material index
        auto IS_SHEET = [&](int m) -> bool
        { return m > 0 && (arma::uword)m <= cols.n_mtl && cols.is_sheet((arma::uword)m); };

        // Thin-sheet case re-derivation. Bit 6 of the hit code is only a hint meaning "at least one of
        // the two faces is a sheet"; codes 69 and 71 are shared by three different topologies on
        // purpose, because this function holds both material indices and can tell them apart while
        // ray_mesh_interact cannot spare the bits. The authoritative case is
        // (code, is_sheet(FBS), is_sheet(SBS), same material), exactly as documented there. Derived
        // once here so the transmission and reflection paths cannot drift apart.
        //   0 = no sheet
        //   1 = isolated sheet, collapsed double-sided duplicate, or a corner whose SBS was dropped
        //   2 = glued entry, host | sheet | solid
        //   3 = glued exit, solid | sheet | air
        //   4 = ordered stack of two different sheets (physics not implemented, see below)
        int sheet_case = 0, sheet_mtl = 0, sheet_solid = 0, sheet_mtl2 = 0;
        if (typeH[6])
        {
            bool shF = IS_SHEET(M1), shS = IS_SHEET(M2);
            if (!typeH[2]) // No co-located second face: nothing can be glued to it
            {
                sheet_case = (shF || shS) ? 1 : 0;
                sheet_mtl = shF ? M1 : (shS ? M2 : 0);
            }
            else if (shF && shS) // Two sheets: a duplicate collapses, a laminate does not
            {
                sheet_case = SAME(M1, M2) ? 1 : 4;
                if (sheet_case == 4)
                {
                    // Traversal order. The layer hit on its front side is traversed first, and that
                    // is what ray_mesh_interact wrote into bit 1; bit 3 records that the normals
                    // gave no order and the geometric FBS -> SBS sequence was used instead. Both
                    // bits survive into this function and only have to be read. The order matters
                    // for the reflect port alone - a stack's transmit coefficient is invariant under
                    // it, by reciprocity - so getting it wrong is silent in transmission.
                    sheet_mtl = typeH[1] ? M1 : M2;
                    sheet_mtl2 = typeH[1] ? M2 : M1;
                }
                else
                    sheet_mtl = M1;
            }
            else if (shF || shS) // One sheet glued to one solid face
            {
                sheet_mtl = shF ? M1 : M2;
                sheet_solid = shF ? M2 : M1;

                // A glued pair whose non-sheet face carries no material has the same medium on both
                // sides of the sheet and is an isolated sheet in disguise. Otherwise bit 1 carries the
                // front/back verdict taken from the solid face, which is the only face that has one.
                sheet_case = (sheet_solid == 0) ? 1 : (typeH[1] ? 2 : 3);
            }
        }

        // Euclidean distance between two points
        auto DIST = [](const double *A, const double *B) -> double
        {
            double a0 = B[0] - A[0], a1 = B[1] - A[1], a2 = B[2] - A[2];
            return std::sqrt(a0 * a0 + a1 * a1 + a2 * a2);
        };

        // Normalize 3-element vector to unit-length
        auto NORMALIZE = [](double *A) -> double
        {
            double len = std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
            if (len > 1.0e-12)
                A[0] /= len, A[1] /= len, A[2] /= len;
            else // Fallback
                A[0] = 1.0, A[1] = 0.0, A[2] = 0.0;
            return len;
        };

        // VBS plane normal: FBS triple, or the SBS triple for an type-(5/23/29) crossing, normalized
        // - type-12 ia a o-i-o and we need to use the normal of the second i-o crossing (the first o-i is pass-through)
        // - type-29 is a ii-o corner crossing, where ray_triangle_intersect outputs the FBS and SBS in order, but the re-launch happens
        //   bast the SBS, so the SBS governs the correct exit.
        double N[3];
        {
            size_t normal_off = ((typeH == 5 || typeH == 23 || typeH == 29) && nH != 0) ? 3 : 0;
            N[0] = (double)p_normal_vecN[ii + (normal_off + 0) * n_rayN];
            N[1] = (double)p_normal_vecN[ii + (normal_off + 1) * n_rayN];
            N[2] = (double)p_normal_vecN[ii + (normal_off + 2) * n_rayN];
            NORMALIZE(N);
        }

        // Segment origin O, destination D and plane crossing F
        const double O[3] = {(double)orig->at(i_ray, 0), (double)orig->at(i_ray, 1), (double)orig->at(i_ray, 2)}; // Ray origin
        const double D[3] = {(double)dest->at(i_ray, 0), (double)dest->at(i_ray, 1), (double)dest->at(i_ray, 2)}; // Ray destination
        double F[3] = {(double)fbsN->at(ii, 0), (double)fbsN->at(ii, 1), (double)fbsN->at(ii, 2)};                // FBS position
        double S[3] = {(double)sbsN->at(ii, 0), (double)sbsN->at(ii, 1), (double)sbsN->at(ii, 2)};                // SBS position

        if (nH == 0) // No-hit fallback
        {
            F[0] = D[0], F[1] = D[1], F[2] = D[2];
            S[0] = D[0], S[1] = D[1], S[2] = D[2];
        }

        // Incoming physical direction
        double dir_prev[3];
        bool use_vbs = false;
        if (p_path_dir_prev)
        {
            dir_prev[0] = (double)p_path_dir_prev[i_ray];
            dir_prev[1] = (double)p_path_dir_prev[i_ray + n_ray];
            dir_prev[2] = (double)p_path_dir_prev[i_ray + 2 * n_ray];
            use_vbs = NORMALIZE(dir_prev) > 1.0e-6;
        }
        if (!use_vbs) // Geometric fallback
        {
            dir_prev[0] = F[0] - O[0];
            dir_prev[1] = F[1] - O[1];
            dir_prev[2] = F[2] - O[2];
            NORMALIZE(dir_prev);
        }

        // Outgoing physical direction
        double dir_next[3];
        if (path_dirN)
        {
            dir_next[0] = (double)p_path_dirN[i_rayN];
            dir_next[1] = (double)p_path_dirN[i_rayN + n_rayN];
            dir_next[2] = (double)p_path_dirN[i_rayN + 2 * n_rayN];
        }
        else if (nH == 0)
            dir_next[0] = F[0] - O[0], dir_next[1] = F[1] - O[1], dir_next[2] = F[2] - O[2];
        else // Geometric fallback
            dir_next[0] = D[0] - F[0], dir_next[1] = D[1] - F[1], dir_next[2] = D[2] - F[2];
        NORMALIZE(dir_next);

        // VBS plane intersection
        // Returns false when the ray is near-parallel to the plane
        auto VBS_INTERSECT = [](const double *O,               // Segment origin O
                                const double *U,               // Physical direction (path_dir), normalized
                                const double *F,               // Plane intersect point (FBS)
                                const double *N,               // Plane normal N, normalized
                                double *dist_OV = nullptr,     // Distance from O to V
                                double *cos_theta_t = nullptr, // |cos(theta_t)| = |d . N|
                                double *V = nullptr) -> bool   // Virtual back-scatter point (VBS)
        {
            double un = U[0] * N[0] + U[1] * N[1] + U[2] * N[2];
            if (std::abs(un) < 1.0e-6)
                return false;

            double s = ((F[0] - O[0]) * N[0] + (F[1] - O[1]) * N[1] + (F[2] - O[2]) * N[2]) / un;
            if (!std::isfinite(s) || s <= 0.0)
                return false;

            if (dist_OV)
                *dist_OV = s;
            if (cos_theta_t)
                *cos_theta_t = std::abs(un);
            if (V)
                V[0] = O[0] + s * U[0], V[1] = O[1] + s * U[1], V[2] = O[2] + s * U[2];
            return true;
        };

        // True in-medium incoming segment and back-side incidence angle, corrected for refraction at
        // the previous entry interaction (VBS, spec 3.1). Default to FBS geometry; overwritten on solve.
        const double dist_orig_fbs = (nH == 0) ? DIST(O, D) : DIST(O, F); // Geometric
        const double dist_fbs_dest = (nH == 0) ? DIST(O, D) : DIST(F, D); // Geometric
        double dist_orig_vbs = dist_orig_fbs;                             // Refracted

        double abs_cos_theta = std::abs(std::cos(theta + 1.570796326794897));
        double theta_t = theta;
        double abs_cos_theta_t = abs_cos_theta;

        // Refine theta_t / d_v from the real incoming direction only when it is supplied. Without
        // path_dir_prev the VBS lands on FBS and the angle stays the fbs_angle default;
        // deriving it from the geometric orig->F direction would override fbs_angle.
        double V[3] = {F[0], F[1], F[2]}; // VBS position
        if (use_vbs)
        {
            double dist_OV, ct;
            if (VBS_INTERSECT(O, dir_prev, F, N, &dist_OV, &ct, V))
            {
                dist_orig_vbs = dist_OV;
                abs_cos_theta_t = (ct > 1.0) ? 1.0 : ct;
                theta_t = std::acos(abs_cos_theta_t) - 1.570796326794897;
                use_vbs = std::abs(abs_cos_theta_t - abs_cos_theta) > 1e-6;
            }
        }

        // Cosine of the angle between the direction the wave has and the segment
        double cos_dir_seg = 1.0;
        if (use_vbs)
        {
            double G[3] = {F[0] - O[0], F[1] - O[1], F[2] - O[2]};
            NORMALIZE(G);
            cos_dir_seg = dir_prev[0] * G[0] + dir_prev[1] * G[1] + dir_prev[2] * G[2];
        }

        // use_vbs says that the VBS is not the FBS: the direction the wave has differs from the
        // segment, so the geometry has to be redone from that direction. Whether the wave was
        // bent at the face behind it is a second question, and with a tracer that bends the
        // segment itself the cosines cannot answer it: there the state does.
        const bool refracted = use_vbs || bent;

        // The bit for a leg that begins at this event by crossing into a medium: the tracer
        // bends, and ray_mesh_interact did bend at this face. Its verdict in the hit code is the
        // one the segment was built on, the ray tube and the rule for a mass-law medium included.
        const int bent_enter = ((interaction_type == 2 || interaction_type == 5) && !total_reflection) ? (int)0x4000 : 0;

        // Accumulated in-medium distance
        double dist_refract = p_acc_refract_in ? (double)p_acc_refract_in[i_ray] : 0.0;
        double dist_geo = p_acc_geo_in ? (double)p_acc_geo_in[i_ray] : 0.0;
        auto DIST_ADD = [&](double val_refract = 0.0, double val_geo = 0.0)
        {
            dist_refract += (val_refract == 0.0) ? dist_orig_vbs : val_refract;
            if (val_geo == 0.0)
                dist_geo += (val_refract == 0.0) ? dist_orig_fbs : val_refract;
            else
                dist_geo += val_geo;
        };
        auto DIST_SET = [&](double val_refract = 0.0, double val_geo = 0.0)
        {
            dist_refract = val_refract;
            dist_geo = (val_geo == 0.0) ? val_refract : val_geo;
        };

        // Wedge test: true when FBS and SBS faces sit at a real angle. No-op (false) when normals are absent or the two faces
        // are a single point. Run only at o-i entries that capture both faces (nH >= 2 types 1/7/13).
        auto fbs_sbs_not_parallel = [&]() -> bool
        {
            if (DIST(F, S) < colocation_dist) // Co-located
                return false;

            double nfx = (double)p_normal_vecN[ii];
            double nfy = (double)p_normal_vecN[ii + n_rayN];
            double nfz = (double)p_normal_vecN[ii + 2 * n_rayN];
            double nsx = (double)p_normal_vecN[ii + 3 * n_rayN];
            double nsy = (double)p_normal_vecN[ii + 4 * n_rayN];
            double nsz = (double)p_normal_vecN[ii + 5 * n_rayN];

            const double tol = 3.8e-3;
            double d = nfx * nsx + nfy * nsy + nfz * nsz;
            return std::abs(d) < 1.0 - tol;
        };

        // Kill the current ray
        auto KILL_RAY = [&]()
        {
            typeR = 0;     // Kill flag (all bits 0)
            if (p_xprmatN) // Set xprmat to 0
                for (size_t iX = 0; iX < nXPR; ++iX)
                    p_xprmatN[ii * nXPR + iX] = (dtype)0.0;
            if (p_gainN) // Set gain to 0
                p_gainN[ii] = (dtype)0.0;
        };

        // Scale current xprmat and gain with a complex factor
        auto SCALE_RAY = [&](double cr, double ci)
        {
            if (p_xprmatN)
                for (size_t iX = 0; iX < nXPR; iX += 2)
                {
                    size_t re_i = ii * nXPR + iX;
                    size_t im_i = re_i + 1;
                    double re = (double)p_xprmatN[re_i];
                    double im = (double)p_xprmatN[im_i];
                    p_xprmatN[re_i] = dtype(re * cr - im * ci);
                    p_xprmatN[im_i] = dtype(re * ci + im * cr);
                }
            if (p_gainN)
                p_gainN[ii] = dtype(double(p_gainN[ii]) * (cr * cr + ci * ci));
        };

        // Replace current xprmat and gain with a fixed gain
        auto REPLACE_BY_GAIN = [&](double gain = 1.0, bool keep_dir = false)
        {
            if (gain == 1.0 && !refl_pass)
            {
                typeR[6] = true;  // Set pass-through flag
                typeR[5] = false; // Clear TIR flag
            }

            if (p_xprmatN)
            {
                for (size_t iX = 0; iX < nXPR; ++iX)
                    p_xprmatN[ii * nXPR + iX] = (dtype)0.0;
                double a = std::sqrt(gain < 0.0 ? 0.0 : gain);
                p_xprmatN[ii * nXPR] = (dtype)a; // VV_re
                if (!is_scalar)
                    p_xprmatN[ii * nXPR + 6] = (dtype)a; // HH_re
            }
            if (p_gainN)
                p_gainN[ii] = (dtype)gain;

            // Transparent pass: carry the incoming physical direction through unchanged,
            // overriding ray_mesh_interact's geometric-based Snell write. Only the caller knows whether
            // dir_prev is the direction to keep, so this is opt-in.
            if (keep_dir && p_path_dirN)
                p_path_dirN[ii] = (dtype)dir_prev[0],
                p_path_dirN[ii + n_rayN] = (dtype)dir_prev[1],
                p_path_dirN[ii + 2 * n_rayN] = (dtype)dir_prev[2];
        };

        // Recover the exact complex in-slab cosine from the traced direction.
        //
        // A traced ray must have a real direction, so qd_refract normalizes a vector built from
        // Re(cos_theta2) and the imaginary part is dropped. Everything downstream then re-derives the
        // boundaries from that real projection - including the ENTRY boundary, whose coefficient
        // ray_mesh_interact had already applied at the true angle with the complex cosine kept. The
        // Airy denominator therefore sums a series against a coefficient the path never carried, and
        // interact_with(slab, air, theta_t) even reconstructs a complex angle inside a lossless
        // medium, which is not an approximation but an impossibility.
        //
        // The information is not lost, only transformed. qd_refract's map from the incidence angle to
        // the traced in-slab cosine,
        //     ct_traced = Re(ct2) / sqrt(eta^2 sin^2(theta_i) + Re(ct2)^2),  ct2 = sqrt(1 - eps sin^2),
        // is deterministic and monotone, and every parameter of it is present here: the entry medium is
        // in the previous-medium word and the slab is the current one. Inverting it returns the exact
        // incidence angle, and the complex cosine follows.
        //
        // Real-index Snell is NOT that map. Inverting that instead - the obvious thing to try - leaves
        // a residual that is worst exactly where the defect is worst, because it ignores the
        // normalization qd_refract performs. So the map itself is inverted: two Newton steps on
        // eta^2 u = t^2 Re(sqrt(1 - eps u))^2 with u = sin^2(theta_i), starting from the real-index
        // guess, which reaches machine precision.
        //
        // Returns 0 where the recovery does not apply, which the resolve reads as "use the projection".
        bool ct_slab_real_inv = false; // set by a solve that returns a cosine; see its end
        auto EXACT_IN_SLAB_COSINE_SOLVE = [&](double ct_traced) -> std::complex<double>
        {
            const std::complex<double> none(0.0, 0.0);
            ct_slab_real_inv = false;

            // No refracted direction and no traced bend, so nothing was projected in the first place.
            if (!refracted || mtl_current < 1)
                return none;

            // The faces are not a parallel pair, so the entry face is not this face's twin and the
            // in-slab angle here says nothing about the angle the ray came in at. Checked here as well
            // as in the caller, because the caller's wedge gate is bypassed at eps == 0.
            if (prev_nonpar)
                return none;

            // The previous-medium word names the medium the current leg was ENTERED from, except
            // after a sheet crossing or a reflection off a sheet, where SHEET_EVENT wrote the sheet
            // into it. The sheet is not the entry medium - Snell is transitive and it drops out -
            // and the medium before it is one level further back than the state reaches.
            //
            // It does not have to be named. With p = e1 sin^2(theta_1) the equation solved below is
            // |p| / |e2| = t^2 Re(sqrt(1 - p / e2))^2, and for a lossless entry medium p is real and
            // e1 appears nowhere else: the recovered cosine is the same for every lossless medium
            // the wave could have come from. So behind a sheet the solve runs with a unit entry
            // permittivity, where u is p itself, and without the bound u <= 1, which belongs to
            // air and not to the invariant. That is exact for a glued entry from air, for the far
            // side of a sheet inside a host, and for the second sheet of a lined wall, however
            // lossy the layers in front are: across parallel faces the transverse wavenumber of
            // a wave that started in air stays real.
            //
            // A lossy solid in the word is the same case wherever the shared face was evaluated
            // at such a wave's cosine, which the transition records in the word (prev_real_inv).
            // Naming the solid there would assume a real ANGLE in it, which that wave does not
            // have. Without the flag the face was evaluated at a real angle in the solid - the
            // wave was launched in it, or its own faces are not parallel - the invariant is the
            // complex e1 sin^2(theta_1), and the named form is the inverse of what was applied.
            const bool behind_sheet = IS_SHEET(mtl_prev);
            const bool unnamed = behind_sheet || (prev_embedded && mtl_prev != 0);
            const double u_max = unnamed ? 1.0e30 : 1.0;

            // The hit code's total-reflection bit does not gate this. It is ray_mesh_interact's
            // verdict on the face AHEAD at the geometric angle, and with an undeviated tracer that
            // is the external angle, so the bit is set from sqrt|eps| sin(theta_ext) >= 1 on a
            // face the wave crosses. What is inverted here is the bend at the face BEHIND. A wave
            // that was not bent there, a leak, holds its geometric direction and was declined
            // above, where the traced and the geometric cosine coincide.
            // Normal incidence is NOT excluded: the projection and the exact value coincide there, but
            // the two propagation factors do not, and declining leaves the hybrid medium_gain form in
            // place while every neighbouring angle uses the exact one. The solve handles it directly -
            // t = 0 gives u = 0 and the first residual is already zero, returning exactly 1.
            if (ct_traced <= 0.0 || ct_traced > 1.0)
                return none;

            Material M_ent = MAT(unnamed ? 0 : mtl_prev), M_slb = MAT(mtl_current);
            std::complex<double> e1 = (M_ent.eta(fGHz) + M_ent.eta_resonance(fGHz)) * M_ent.mu(fGHz);
            std::complex<double> e2 = (M_slb.eta(fGHz) + M_slb.eta_resonance(fGHz)) * M_slb.mu(fGHz);
            if (std::abs(e2) <= 0.0)
                return none;
            std::complex<double> eps_r = e1 / e2; // qd_refract's eta1_div_eta2
            double eta2 = std::abs(eps_r);        // qd_refract's eta, squared
            if (eta2 <= 0.0 || std::abs(eps_r - 1.0) < 1.0e-12)
                return none; // matched media: no bend to invert

            double ct = ct_traced;
            double ct2c = 1.0 - ct * ct;
            double t = std::sqrt((ct2c > 0.0) ? ct2c : 0.0) / ct; // tan of the traced in-slab angle
            double u = t * t / (eta2 * (1.0 + t * t));            // real-index Snell start
            u = (u > u_max) ? u_max : u;

            // Newton on the map, until the step is below rounding. Three steps reach that from the
            // real-index start wherever real-index Snell has a solution near the root; in a lossy
            // medium rarer than the one the wave came from it has none past sqrt|eps_r| sin = 1,
            // and towards an invariant above one the start is far as well.
            for (int k = 0; k < 50; ++k)
            {
                std::complex<double> w = std::sqrt(1.0 - eps_r * u);
                double fu = eta2 * u - t * t * std::real(w) * std::real(w);
                double dfu = eta2 + t * t * std::real(w) * std::real(eps_r / w);
                if (!(std::abs(dfu) > 1.0e-300))
                    break;
                double du = fu / dfu;
                u -= du;
                u = (u < 0.0) ? 0.0 : ((u > u_max) ? u_max : u);
                if (k >= 2 && std::abs(du) <= 1.0e-15 * std::abs(u))
                    break;
            }

            // Whether the wave in this medium has a real transverse invariant and a complex
            // cosine: solved for a real invariant, or for a real angle in a lossless entry
            // medium, which is the same statement, and the medium itself lossy. A transition out
            // of this leg then says so in the word it writes. Decided on the media and the state,
            // never on a value, so it is the same at every frequency of a run.
            ct_slab_real_inv = (unnamed || std::imag(e1) == 0.0) && std::imag(e2) != 0.0;

            std::complex<double> arg = 1.0 - eps_r * u;
            if (qd_evanescent(arg))
                arg = std::complex<double>(std::real(arg), -0.0); // decaying evanescent root
            return std::sqrt(arg);
        };

        // Three sites want the same value - the exit interface, the slab series and the traversal
        // factor - and every gate above depends on the ray, not on the caller. mtl_current is the
        // one thing that moves between them (APPLY_TRANSITION rewrites it before the interface
        // call), so it is the memo key; a solve is one Newton triple and this keeps it at one per
        // medium per ray.
        int ct_slab_exact_for = -2; // -2 = nothing solved yet (0 is a valid medium: air)
        std::complex<double> ct_slab_exact_val(0.0, 0.0);
        auto EXACT_IN_SLAB_COSINE = [&]() -> std::complex<double>
        {
            if (ct_slab_exact_for != mtl_current)
                ct_slab_exact_val = EXACT_IN_SLAB_COSINE_SOLVE(abs_cos_theta_t), ct_slab_exact_for = mtl_current;
            return ct_slab_exact_val;
        };

        // VBS-corrected coefficients
        std::complex<double> cTE(0.0, 0.0), cTM(0.0, 0.0);

        // True where REPLACE_BY_VBS evaluated its interface at the recovered cosine of a wave with
        // a real transverse invariant in a lossy medium. A transition writes it into the word.
        bool face_real_inv = false;

        // The geometry a direction out of a lossy medium owes.
        //
        // With a refracting type ray_mesh_interact built the next segment, and the three vertex
        // rays of a tube, by refracting at a real angle in the medium the ray leaves. Where the
        // direction written here is the one of a wave whose cosine in that medium was recovered
        // for a real transverse invariant, the segment has to follow it: origN and destN are put
        // on one line through the point the ray leaves at, the SBS of a co-located pair and the
        // FBS otherwise, along dir_next. Each vertex ray is a ray of its own and gets the same
        // construction from its own direction in the medium: its recovered cosine, the Snell
        // ratio with the length the projection lost, the far cosine of the pair. Its footprint
        // on the face is where it arrived and does not move.
        //
        // Called by the writers of such a direction, right where they write it, so the state the
        // recovery reads is the one the spine was solved in. All or nothing: a vertex that cannot
        // be formed leaves the ray with the geometry it had. A leak behind a closed face gets the
        // same construction with air as the far medium, out of a lossless medium as well. An
        // open face out of a lossless medium, a recovery that declines and every other type
        // never get here.
        auto RELAUNCH = [&](const Material &A, const Material &B, int itype)
        {
            if (!p_origN || (interaction_type != 2 && interaction_type != 5))
                return;

            double td[9] = {};
            if (p_tridirN)
            {
                const bool keep_real_inv = ct_slab_real_inv; // the solve below rewrites it
                bool ok = true;
                for (int iTube = 0; iTube < 3 && ok; ++iTube)
                {
                    double v[3] = {(double)p_tridir[i_ray + (3 * iTube) * n_ray],
                                   (double)p_tridir[i_ray + (3 * iTube + 1) * n_ray],
                                   (double)p_tridir[i_ray + (3 * iTube + 2) * n_ray]};
                    NORMALIZE(v);
                    double cv = std::abs(v[0] * N[0] + v[1] * N[1] + v[2] * N[2]);
                    cv = (cv > 1.0) ? 1.0 : cv;

                    std::complex<double> c_a = EXACT_IN_SLAB_COSINE_SOLVE(cv), c_b;
                    ok = std::abs(c_a) > 0.0;
                    if (!ok)
                        break;

                    double sn = 1.0;
                    bool t_ab = false;
                    A.interact_with(B, itype, std::acos(cv) - 1.570796326794897, fGHz, nullptr, nullptr, &c_b, nullptr, &sn, &t_ab,
                                    false, &c_a, false);
                    ok = !t_ab && std::real(c_b) > 0.0;
                    if (!ok)
                        break;

                    sn *= std::sqrt(std::abs(1.0 - c_a * c_a) + std::real(c_a) * std::real(c_a));
                    qd_refract(v[0], v[1], v[2], N[0], N[1], N[2], sn, cv, c_b, td[3 * iTube], td[3 * iTube + 1], td[3 * iTube + 2]);
                }
                ct_slab_real_inv = keep_real_inv;
                if (!ok)
                    return;
            }

            const double *A0 = (nH >= 2 && DIST(F, S) < colocation_dist) ? S : F; // where ray_mesh_interact relaunched from
            const double Dn[3] = {(double)p_destN[ii], (double)p_destN[ii + n_rayN], (double)p_destN[ii + 2 * n_rayN]};
            const double len = DIST(F, Dn), off = qd_relaunch_offset(F[0], F[1], F[2]);
            for (size_t c = 0; c < 3; ++c)
            {
                p_origN[ii + c * n_rayN] = (dtype)(A0[c] + off * dir_next[c]);
                p_destN[ii + c * n_rayN] = (dtype)(A0[c] + len * dir_next[c]);
            }
            if (p_tridirN)
                for (size_t c = 0; c < 9; ++c)
                    p_tridirN[ii + c * n_rayN] = (dtype)td[c];
        };

        // Replace the FBS-relative interface result ray_mesh_interact wrote with the VBS-equivalent at theta_t
        // - interface coefficients (interface_gain folded inside interact_with), incoming basis from dir_in,
        //   outgoing basis from the corrected continuation. In-medium magnitude/phase stay deferred to close_med.
        // - geom: 0 = reflect at VBS, 2 = Snell refract at VBS, 3 = exit into air. A face closed at the VBS angle hands on the leak's direction.
        auto REPLACE_BY_VBS = [&](int Ma, int Mb, int itype, int geom) -> double
        {
            // The direction an exit into air owes, where it can be formed.
            //
            // ray_mesh_interact wrote the direction for this face from the geometric segment, and
            // with an undeviated tracer that segment runs at the external angle, so its Snell bend
            // out of the medium is applied to a direction the wave does not have: out of eps 1.50
            // a ray that came in at 40 deg left at 51.9 deg. What bends is dir_prev. Across parallel
            // faces the transverse invariant p of a wave from air is real and conserved, and the
            // direction in air is sqrt(p) against sqrt(1 - p), parallel to the one it came in at.
            // That is the construction a transition uses: the Snell ratio with the length the
            // projection lost, and the far cosine at the recovered cosine.
            //
            // Out of a lossy medium whose cosine was recovered for a real invariant that is exact.
            // Where the recovery declines the Snell ratio and the far cosine are the real-angle
            // ones, which is the wave for a lossless medium and the map a glued exit applies for
            // a lossy one. A leg with no refracted direction keeps what ray_mesh_interact wrote,
            // and a face that is closed hands on the direction of the leak, below.
            auto EXIT_DIRECTION = [&](const Material &A, const Material &B, bool real_inv, const std::complex<double> &ct_a,
                                      double snell_ab, const std::complex<double> &cos_b, bool tir_ab, bool leak = false) -> bool
            {
                if (!refracted || tir_ab || !(std::real(cos_b) > 0.0))
                    return false;
                double snell_dir = snell_ab;
                if (real_inv)
                    snell_dir *= std::sqrt(std::abs(1.0 - ct_a * ct_a) + std::real(ct_a) * std::real(ct_a));
                qd_refract(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], snell_dir, abs_cos_theta_t, cos_b,
                           dir_next[0], dir_next[1], dir_next[2]);
                if (p_path_dirN)
                    p_path_dirN[ii] = (dtype)dir_next[0],
                    p_path_dirN[ii + n_rayN] = (dtype)dir_next[1],
                    p_path_dirN[ii + 2 * n_rayN] = (dtype)dir_next[2];
                if (real_inv || leak) // out of a lossy medium, and a leak out of any: the segment follows
                    RELAUNCH(A, B, itype);
                return true;
            };

            // A resolved ray is evaluated as any other: what ray_mesh_interact wrote for this face
            // was formed at the angle of the segment. Only the series is skipped for it, by the
            // caller.

            // Compute coefficients (cTE and cTM are required by SLAB_AIRY_FACTOR, so we always compute them)
            // only cTE/cTM are needed past the fast return; the geometry outs are used only in the VBS branch
            std::complex<double> cos_t2, e1de2;
            double xprmat[8], pgain;
            double snell = 1.0;
            bool tir = false;

            // This interface is met from INSIDE the current medium. Where the ray reached it by
            // refracting through a parallel face, that medium's propagation cosine is complex and
            // the traced direction carries only its real projection; the recovery inverts qd_refract
            // and returns the value the entry boundary was evaluated at. Handing it over here is
            // what makes the entry and exit obliquity factors cancel: both are then formed at the
            // same cosine in the shared medium, so the pair telescopes to
            // Re(z_far c_far)/Re(z_entry c_entry) - exactly 1 for a solid re-entering the medium it
            // came from - instead of leaving the ratio of two different views of the same flux.
            //
            // The gates are the recovery's own, so where it declines the cosine is zero and the real
            // angle is used exactly as before. Ma is mtl_current at both call sites; the test is
            // there so the value can never be read as a cosine in some other medium.
            //
            // cos_t2 moves with it. Snell is transitive, so the exact cosine turns the refraction
            // root into the cosine the OUTER pair of media gives, which is the one the continuation
            // direction owes. Only geom 2 consumes it - an exit keeps the traced direction and a
            // mirror does not use it at all.
            std::complex<double> ct_in = (Ma == mtl_current) ? EXACT_IN_SLAB_COSINE()
                                                             : std::complex<double>(0.0, 0.0);
            bool use_exact_ct = std::abs(ct_in) > 0.0;
            face_real_inv = use_exact_ct && ct_slab_real_inv;

            // Without a refracted direction the wave is held at its geometric angle, as it was in
            // ray_mesh_interact, and the two have to give the same verdict.
            MAT(Ma).interact_with(MAT(Mb), itype, theta_t, fGHz, &cTE, &cTM, &cos_t2, &e1de2, &snell, &tir,
                                  false, use_exact_ct ? &ct_in : nullptr, !refracted);
            typeR[5] = tir; // TIR re-evaluated at the corrected VBS angle (overrides the geometric-angle verdict)

            // The direction a closed face hands on.
            //
            // What crosses a closed face is a leak, and a leak carries the transverse wavenumber
            // and nothing else of the wave in this medium. Where that wavenumber has a direction
            // in air, an invariant below one, that is the direction: for a wave that came from
            // air through parallel faces the one it arrived in, at every frequency, which is what
            // the same layer as a sheet hands the wall behind it. It is the direction an exit
            // into air writes, so it is written by that construction with air as the far medium.
            // Where it has none - an invariant of one or more, a leg with no refracted
            // direction - the face hands on the direction the wave has.
            auto LEAK_DIRECTION = [&]()
            {
                dir_next[0] = dir_prev[0], dir_next[1] = dir_prev[1], dir_next[2] = dir_prev[2];
                if (Mb == 0)
                    return;
                std::complex<double> cos_air;
                double snell_air = 1.0;
                bool tir_air = false;
                MAT(Ma).interact_with(MAT(0), itype, theta_t, fGHz, nullptr, nullptr, &cos_air, nullptr, &snell_air, &tir_air,
                                      false, use_exact_ct ? &ct_in : nullptr, false);
                EXIT_DIRECTION(MAT(Ma), MAT(0), face_real_inv, ct_in, snell_air, cos_air, tir_air, true);
            };

            // Transition into buffer
            if (!use_vbs && mtl_buffer && Mb == mtl_buffer && !SAME(mtl_buffer, M1))
            {
                qd_polbasis(dir_prev[0], dir_prev[1], dir_prev[2], dir_next[0], dir_next[1], dir_next[2],
                            N[0], N[1], N[2], 1.0, cTE, cTM, is_scalar, xprmat, pgain);
                if (p_xprmatN)
                    for (size_t iX = 0; iX < nXPR; ++iX)
                        p_xprmatN[ii * nXPR + iX] = (dtype)xprmat[iX];
                if (p_gainN)
                    p_gainN[ii] = (dtype)pgain;
            }

            if (!refracted) // Fast return
                return dist_fbs_dest;

            // A bent leg between two lossless media: the segment is the wave, and what
            // ray_mesh_interact wrote from it is this face at this angle. Left as it stands.
            // A mass-law medium is not, because ray_mesh_interact held the wave undeviated in it.
            if (!use_vbs)
            {
                Material A = MAT(Ma), B = MAT(Mb);
                if (std::imag((A.eta(fGHz) + A.eta_resonance(fGHz)) * A.mu(fGHz)) == 0.0 &&
                    std::imag((B.eta(fGHz) + B.eta_resonance(fGHz)) * B.mu(fGHz)) == 0.0 && !(A.m > 0.0) && !(B.m > 0.0))
                {
                    // The direction an exit writes is the one dir_prev refracts to, as for any
                    // other leg: the segment carries the relaunch offset of the event before it.
                    if (geom == 3)
                        EXIT_DIRECTION(A, B, false, ct_in, snell, cos_t2, tir);
                    else if (geom == 2 && tir) // closed: the segment is the wave in A, the leak is not
                    {
                        LEAK_DIRECTION();
                        if (p_path_dirN)
                            p_path_dirN[ii] = (dtype)dir_next[0],
                            p_path_dirN[ii + n_rayN] = (dtype)dir_next[1],
                            p_path_dirN[ii + 2 * n_rayN] = (dtype)dir_next[2];
                    }
                    return dist_fbs_dest;
                }
            }

            if (use_vbs)
                typeR[1] = true; // Set VBS correction flag

            // Update the outgoing direction based on the VBS geometry (using the refracted incoming direction)
            if (geom == 0) // Reflect at VBS
                qd_reflect(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], dir_next[0], dir_next[1], dir_next[2]);
            else if (geom == 2 && !tir) // Snell refract at VBS (undeviated under TIR)
            {
                // The direction in the next medium has to be the one a single refraction from
                // air writes, tangential sqrt(p / |e2|) against Re(cos_t2) with p the transverse
                // invariant, because that is the map the recovery inverts there and the wave in
                // that medium does not depend on what lies in front of it. dir_prev is itself a
                // projection: qd_refract normalized a vector of length
                // sqrt(|sin^2| + Re(cos)^2) in this medium, which is 1 only for a real cosine,
                // so its tangential part is short by that length. Putting the length back on the
                // Snell ratio removes it, and no direction further on inherits it.
                double snell_dir = snell;
                if (face_real_inv)
                    snell_dir *= std::sqrt(std::abs(1.0 - ct_in * ct_in) + std::real(ct_in) * std::real(ct_in));
                qd_refract(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], snell_dir, abs_cos_theta_t, cos_t2, dir_next[0], dir_next[1], dir_next[2]);
                if (face_real_inv) // out of a lossy solid: the segment in the next medium follows
                    RELAUNCH(MAT(Ma), MAT(Mb), itype);
            }
            else if (tir) // closed at this angle: no Snell direction, the leak keeps its transverse wavenumber
                LEAK_DIRECTION();
            else if (geom == 3) // exit into air: the direction of dir_prev refracted, where it is exact
                EXIT_DIRECTION(MAT(Ma), MAT(Mb), face_real_inv, ct_in, snell, cos_t2, tir);

            if (p_path_dirN && (geom != 3 || tir)) // write the corrected continuation; an open exit wrote its own or keeps the traced one
                p_path_dirN[ii] = (dtype)dir_next[0],
                p_path_dirN[ii + n_rayN] = (dtype)dir_next[1],
                p_path_dirN[ii + 2 * n_rayN] = (dtype)dir_next[2];

            qd_polbasis(dir_prev[0], dir_prev[1], dir_prev[2], dir_next[0], dir_next[1], dir_next[2],
                        N[0], N[1], N[2], 1.0, cTE, cTM, is_scalar, xprmat, pgain);
            if (p_xprmatN)
                for (size_t iX = 0; iX < nXPR; ++iX)
                    p_xprmatN[ii * nXPR + iX] = (dtype)xprmat[iX];
            if (p_gainN)
                p_gainN[ii] = (dtype)pgain;

            // Return virtual vbs-dest distance using the VBS position as start
            // The refracted angle at the VBS differs from the FBS angle, so the refracted path length
            // in the entered medium differs as well.
            double dist_VV; // Distance from VBS to virtual destination
            if (VBS_INTERSECT(V, dir_next, D, N, &dist_VV))
                return dist_VV;
            return dist_fbs_dest; // Fallback
        };

        // Buffer word for a leg that begins by crossing out of a mass-law solid whose closed leg
        // had the surface-mass surrogate mass_n: the flag and the thickness code. Zero where the
        // body carried no mass term, so that it is absent from the plate as it was from the path.
        auto MASS_CODE = [](double mass_n) -> short
        {
            if (!(mass_n > 0.0))
                return (short)0;
            double k = 1.0 + std::round((std::log10(mass_n) + 4.0) * (32766.0 / 6.0));
            k = (k < 1.0) ? 1.0 : ((k > 32767.0) ? 32767.0 : k);
            return (short)((int)0x8000 | (int)k);
        };

        // Summed X0 of the bonded mass-law bodies crossed on the way into the current leg, as far
        // as the state names them. A solid in the previous-medium word is one body, its surface
        // mass decoded from the buffer word. A sheet there is one body by its thickness column,
        // and where it lies inside the current medium the part of that medium in front of it is a
        // second one, of the current material and the decoded thickness.
        auto MASS_RUN = [&]() -> double
        {
            if (!mass_crossed)
                return 0.0;
            Material P = MAT(mtl_prev);
            if (!(P.m > 0.0))
                return 0.0;
            double d_code = (mass_code < 1) ? 0.0 : std::pow(10.0, -4.0 + 6.0 * (double)(mass_code - 1) / 32766.0);
            if (IS_SHEET(mtl_prev))
            {
                double X = P.mass_X0(P.thk, fGHz);
                if (prev_embedded && d_code > 0.0)
                {
                    Material H = MAT(mtl_current);
                    if (H.m > 0.0)
                        X += H.mass_X0(d_code, fGHz);
                }
                return X;
            }
            if (mass_code < 1)
                return 0.0;
            return P.mass_X0(d_code, fGHz);
        };

        // Normal thickness of material R that has the X0 of a leg of R with normal thickness d
        // together with the bonded bodies crossed before it, as far as the state names them and
        // they have R's slope m. With one slope, X0 = ((f / fRef) d)^(m / 20) of every body has
        // the same power of the frequency, so the summed X0 is that of one body of R at every
        // frequency: each thickness is scaled by fRef_R / fRef, and the m / 20 powers add. No
        // frequency enters, which is what lets the result travel as the thickness code. A body of
        // another slope has no such thickness and is left out, as every body behind the one the
        // state names was before. extra is the first layer of an ordered stack, which the
        // previous-medium word will not hold.
        auto MASS_FOLD = [&](int R, double d, int extra = 0) -> double
        {
            Material Rm = MAT(R);
            double p = 0.05 * Rm.m, s = std::pow(d, p);
            auto ADD = [&](const Material &B, double thk)
            {
                if (B.m == Rm.m && thk > 0.0)
                    s += std::pow(thk * Rm.fRef / B.fRef, p);
            };
            if (extra != 0)
                ADD(MAT(extra), MAT(extra).thk);
            if (mass_crossed)
            {
                Material P = MAT(mtl_prev);
                double d_code = (mass_code < 1) ? 0.0 : std::pow(10.0, -4.0 + 6.0 * (double)(mass_code - 1) / 32766.0);
                if (!IS_SHEET(mtl_prev))
                    ADD(P, d_code);
                else
                {
                    ADD(P, P.thk);
                    if (prev_embedded)
                        ADD(MAT(mtl_current), d_code);
                }
            }
            return std::pow(s, 1.0 / p);
        };

        // Apply medium gain and phase
        auto SCALE_BY_MEDIUM = [&](int m)
        {
            Material mtl = MAT(m);
            double n_re = std::real(std::sqrt(mtl.eta(fGHz) * mtl.mu(fGHz)));
            double k0 = 2.0 * 3.14159265358979323846 * fGHz * 1e9 / 299792458.0;

            // Walk-off (cos^2) only for a wave that was not bent, a leak or a leg with no refracted
            // direction, whose polyline is the wave. The free-space reference sits on the same
            // thickness axis, so dist_geo carries the geometric-incidence cos^2. A wave that was
            // bent takes the forms below on every pass, the reflect pass and a resolved ray
            // included, so that one leg is one form whichever event closes it.
            bool walk_off = !refracted;

            // Which point the phase belongs to: the one the polyline leaves at.
            //
            // The path length, the delay and the free-space phase of a path are taken from the
            // polyline, one polyline for every frequency of a run, so the coefficient owes the
            // field where that polyline leaves the medium. The wave itself leaves beside it
            // wherever the polyline is not the refracted ray: with the undeviated tracer always,
            // with the refracting one at every frequency but the base one. For a plane wave the
            // two points differ by the transverse wavenumber times the lateral distance between
            // them, in phase only. Nothing of the geometry moves for it; use_vbs says that the
            // two differ, and where it is clear the forms below are what they were, bit for bit.

            // Exact in-medium propagation, wherever the recovered cosine exists.
            //
            // medium_gain takes the ITU-R P.2040 loss rate along the PATH, which is the decay of a
            // homogeneous plane wave. The wave inside a traced solid is not homogeneous: entered
            // from a lossless medium its transverse wavenumber is real, so the whole decay sits on
            // the normal component and is exp(k0 Im(n cs) d) over the NORMAL thickness, independent
            // of how long the ray is. The two agree only at normal incidence, which is where the
            // 2.5e-5 dB figure for the hybrid form was taken; at 85 deg through 100 mm of
            // eps 1.50 - 0.40j the same substitution is worth 0.040 dB, and it is the entire
            // residual left between a traced solid and the analytic slab once the interfaces are
            // right.
            //
            // Written as one factor, exp(-j k0 (n cs d + q s)), with d the normal thickness and s
            // the transverse displacement the tracer walked. q is the transverse wavenumber the
            // medium carries, q^2 = n^2 (1 - cs^2), which Snell conserves from the entry medium. For
            // a real index the two terms recombine into n * dist_refract exactly, so this reduces to
            // the form below rather than replacing it. The normal thickness is pure geometry - the
            // traced path times the traced cosine - and only the phase and loss across it use the
            // recovered angle; qd_slab_resolve splits it the same way.
            //
            // The conductivity lives inside n, so only the alpha column is drawn from the material.
            // Mass-law materials are excluded: their eta is calibrated for surface impedance, the
            // in-medium index is not physical, and the mass term is not a per-meter loss at all.
            // Where the tracer did not trace the bend the ledger is not on a refracted path and
            // there is nothing to be exact about, so the walk-off branch keeps the form it had.
            std::complex<double> cs = (!walk_off && m == mtl_current && !(mtl.m > 0.0))
                                          ? EXACT_IN_SLAB_COSINE()
                                          : std::complex<double>(0.0, 0.0);

            if (std::abs(cs) > 0.0)
            {
                std::complex<double> n_cx = std::sqrt(mtl.eta(fGHz) * mtl.mu(fGHz)); // resonance-excluded
                double d_norm = dist_refract * abs_cos_theta_t;                      // normal thickness
                double sin_t = 1.0 - abs_cos_theta_t * abs_cos_theta_t;
                double s_lat = dist_refract * std::sqrt((sin_t > 0.0) ? sin_t : 0.0); // transverse walk

                std::complex<double> q_arg = n_cx * n_cx * (1.0 - cs * cs);
                if (qd_evanescent(q_arg))
                    q_arg = std::complex<double>(std::real(q_arg), -0.0); // decaying evanescent root
                std::complex<double> q = std::sqrt(q_arg);

                double ex = mtl.excess_gain(dist_refract, fGHz); // alpha only, no conductivity
                std::complex<double> P =
                    std::exp(std::complex<double>(0.0, -1.0) * k0 * (n_cx * cs * d_norm + q * s_lat)) *
                    std::sqrt((ex < 0.0) ? 0.0 : ex);

                // From the point the refracted ray leaves at to the point the polyline leaves at:
                // the lateral walk of the polyline in place of the wave's. A phase, also where the
                // invariant is complex, so that every magnitude stays the one of the ray.
                if (use_vbs)
                {
                    double sin_i = 1.0 - abs_cos_theta * abs_cos_theta;
                    double s_poly = dist_geo * std::sqrt((sin_i > 0.0) ? sin_i : 0.0);
                    P *= std::exp(std::complex<double>(0.0, -1.0) * k0 * std::real(q) * (s_poly - s_lat));
                }

                // Free-space reference over the geometric length, as below
                P *= std::exp(std::complex<double>(0.0, 1.0) * k0 * dist_geo);
                SCALE_RAY(std::real(P), std::imag(P));
                return;
            }

            // A leg entered through a bonded mass-law body pays what the plate is worth above
            // what that body paid; see Material::medium_gain.
            double g = mtl.medium_gain(dist_refract, fGHz, abs_cos_theta_t, dist_geo, abs_cos_theta,
                                       0.0015, -1.0, -1.0, (mtl.m > 0.0) ? MASS_RUN() : 0.0);
            double cos2 = walk_off ? abs_cos_theta_t * abs_cos_theta_t : 1.0; // medium side, theta_t
            double cos2_geo = walk_off ? abs_cos_theta * abs_cos_theta : 1.0; // free-space side, theta_i
            double excess_phase = k0 * (n_re * dist_refract * cos2 - dist_geo * cos2_geo);

            // A bent wave with no recovered cosine, in a mass-law solid or where the recovery
            // declines: the phase of the real index at the point the polyline leaves at, which is
            // the wave vector along dir_prev times the polyline.
            if (!walk_off && use_vbs)
                excess_phase = k0 * (n_re * dist_geo * cos_dir_seg - dist_geo);

            double amp = std::sqrt(g < 0.0 ? 0.0 : g);
            SCALE_RAY(amp * std::cos(-excess_phase), amp * std::sin(-excess_phase));
        };

        // Obtain the material of the current medium by its exit face, flag a mismatch with mtl_current
        auto GET_EXIT_MATERIAL = [&]() -> int
        {
            int mtl_exit = 0;
            if (typeH == 1 || typeH == 5 || typeH == 13 || typeH == 21 || typeH == 29) // i-o or i-i
                mtl_exit = M1;
            else if (typeH == 7) // i-i with M2 (next, front) hit first
                mtl_exit = M2;

            if (mtl_exit && mtl_current != mtl_exit) // Current and exit mismatch
                typeR[4] = true;                     // Set fix-flag

            return mtl_exit;
        };

        // Analytic thin-slab (Fabry-Perot) resolution, applied to the current medium.
        // The physics lives in qd_slab_resolve; what stays here is the index lookup, the wedge gate
        // and the buffer write, because the two public callers write to different buffers.
        // The returned factor folds the per-polarization Airy phase into xprmat so TE and TM carry
        // their own slab retardation (and the resulting depolarization) while the total power gain
        // stays at the TE/TM-averaged value.
        // Argument order is historical: mtl_near is the interface ahead of the ray, mtl_far the one
        // it entered through - the reverse of qd_slab_resolve's incidence-side convention. Correction
        // mode is symmetric in the two so this is harmless, but a produce-mode call from here would
        // have to swap them.
        // Returns true when the series resolved (caller sets any extra state, e.g. the reflected slab flag).
        bool series_summed = false; // this event summed a bounce series; the verdict handed out
        auto SLAB_AIRY_FACTOR = [&](int mtl_slab, int mtl_near, int mtl_far, int mtl_back = 0) -> bool
        {
            // The verdict of this event, where the caller hands it in: what the base frequency
            // of a run decided. The paths are the base frequency's, so every other frequency
            // sums where it summed and leaves the bounces to the tracer where it did not.
            const bool given = p_verdict_inN != nullptr;
            if (given && p_verdict_inN[ii] == 0)
                return false;

            if (!given && eps > 0.0 && prev_nonpar) // known wedge/edge -> re-emit (eps = 0.0 always resolves)
                return false;

            // What lies behind a sheet named as mtl_near, where the caller knows it. SHEET_EVENT
            // does, because at the crossing it still holds the medium on the far side of the sheet;
            // on any later leg the previous-medium word holds the sheet and what is behind it is one
            // level further back than the state format reaches, so air is assumed. Air is also the
            // default here, which is why every other call site passes nothing.
            Material M_back = MAT(mtl_back);

            // What lies behind a sheet named as mtl_far, the mirror behind the ray. Every caller
            // passes the previous-medium word there. A sheet in it that lies inside the slab is
            // backed by the slab; a lining is backed by air, which is the default.
            Material M_back_far = MAT(mtl_slab);
            bool far_in_slab = prev_embedded && mtl_far == mtl_prev && IS_SHEET(mtl_far);

            std::complex<double> fac_te, fac_tm;
            if (!qd_slab_resolve(MAT(mtl_near), MAT(mtl_slab), MAT(mtl_far),
                                 dist_refract, dist_geo, abs_cos_theta_t, abs_cos_theta,
                                 fGHz, is_scalar, eps, false, &fac_te, &fac_tm,
                                 nullptr, nullptr, false, -1.0, EXACT_IN_SLAB_COSINE(),
                                 std::complex<double>(0.0, 0.0), &M_back, false, false, 0.0,
                                 far_in_slab ? &M_back_far : nullptr, given))
                return false;
            series_summed = true;

            // Scalar mode carries one port and the factor is a pure scale, so scale the stored
            // xprmat in place. Rebuilding it from cTE would round-trip through dtype differently.
            if (is_scalar)
            {
                SCALE_RAY(std::real(fac_te), std::imag(fac_te));
                return true;
            }

            // Rebuild xprmat from the single-pass coeffs with the per-channel slab factor applied to
            // cTE/cTM before the polarization basis. gainN follows from the rebuilt matrix.
            double xprmat[8], pgain;
            qd_polbasis(dir_prev[0], dir_prev[1], dir_prev[2], dir_next[0], dir_next[1], dir_next[2],
                        N[0], N[1], N[2], 1.0, cTE * fac_te, cTM * fac_tm, false, xprmat, pgain);
            if (p_xprmatN)
                for (size_t iX = 0; iX < nXPR; ++iX)
                    p_xprmatN[ii * nXPR + iX] = (dtype)xprmat[iX];
            if (p_gainN)
                p_gainN[ii] = (dtype)pgain;
            return true;
        };

        // Re-resolve a thin-sheet crossing against the media the tracked state reports.
        //
        // ray_mesh_interact derives the sheet's near and far media from face geometry alone, so it has
        // to assume air on any side the mesh does not name. That assumption holds only while the ray is
        // in air. A sheet inside a medium is bounded by that medium on both sides; a sheet glued to a
        // solid but reached from inside another medium has that medium on its near side. Getting it
        // wrong is not a small error: it moves both Fresnel boundaries, the Airy denominator that
        // couples them, and the lumped interface gain, all at once. This function is the first place
        // that knows the answer, which is why qd_slab_resolve sits at file scope instead of being a
        // lambda inside ray_mesh_interact.
        //
        // The incidence angle is theta_t rather than the FBS angle, so the redo also carries the same
        // geometric correction that REPLACE_BY_VBS gives a traced solid. That makes VBS correctness
        // unconditional here, which is why bit 1 of the resolved type is free to mean something else
        // inside the sheet region.
        //
        // Sets cTE / cTM so that a following SLAB_AIRY_FACTOR can fold the host medium's own series
        // into the same TE/TM pair. Returns false when the composite must be left exactly as
        // ray_mesh_interact wrote it.
        auto SHEET_RESOLVE = [&](int mtl_slab, int mtl_near, int mtl_far, bool disp_far,
                                 int mtl_slab2 = 0, double mass_bonded = 0.0) -> bool
        {
            Material M_slab = MAT(mtl_slab), M_near = MAT(mtl_near), M_far = MAT(mtl_far);
            Material M_slab2 = MAT(mtl_slab2);
            double thk = M_slab.thk;
            double thk2 = (mtl_slab2 != 0) ? M_slab2.thk : 0.0;
            if (thk <= 0.0) // Not a sheet material after all: leave the composite alone
                return false;
            if (mtl_slab2 != 0 && thk2 <= 0.0) // an ordered stack whose second layer is not a sheet
                return false;

            // Geometry and the total-reflection verdict from the outer pair, at theta_t.
            //
            // Both have to be re-derived here rather than taken from the hit code. With an undeviated
            // tracer a ray inside a solid travels at its external angle, so ray_mesh_interact evaluated
            // the crossing at an angle that is not the in-slab one, and past the internal critical angle
            // it declares a mirror where there is none - a glued exit then loses tens of dB.
            // REPLACE_BY_VBS performs exactly this correction for solids and this path deliberately does
            // not call it. Only the outer media decide: Snell is transitive, so the sheet can neither
            // create nor remove total reflection, and it cannot bend the outgoing ray either.
            std::complex<double> ct2_far;
            double snell = 1.0;
            bool tir = false;
            M_near.interact_with(M_far, interaction_type, theta_t, fGHz, nullptr, nullptr,
                                 &ct2_far, nullptr, &snell, &tir, false, nullptr, !refracted);

            // Where the near medium's cosine was recovered, the verdict and the far cosine are
            // the ones at that cosine, as at a bare face. For a lossy near medium the real-angle
            // test reads the invariant short by the length the projection lost, and finds the
            // pair open on a wave that is closed.
            std::complex<double> c_n(0.0, 0.0), c_f(0.0, 0.0);
            if (mtl_near == mtl_current && mtl_far != mtl_near)
            {
                c_n = EXACT_IN_SLAB_COSINE();
                if (std::abs(c_n) > 0.0)
                    M_near.interact_with(M_far, interaction_type, theta_t, fGHz, nullptr, nullptr, &c_f, nullptr,
                                         nullptr, &tir, false, &c_n, false);
            }
            typeR[5] = tir;

            // Continuation direction. Leaving path_dirN as ray_mesh_interact wrote it replaces the
            // physical refracted direction with the geometric one, which silently breaks the VBS chain
            // for every later interaction on the path: the next face is then evaluated at the wrong
            // angle and, inside a solid, total-reflects. With near == far the interface call returns
            // eta = 1 and a refraction cosine equal to the incidence cosine, so the refraction branch
            // reproduces dir_prev and the isolated sheet needs no special case.
            if (refl_pass)
                qd_reflect(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2],
                           dir_next[0], dir_next[1], dir_next[2]);
            else if (tir) // no Snell direction under total reflection; any leak travels undeviated
                dir_next[0] = dir_prev[0], dir_next[1] = dir_prev[1], dir_next[2] = dir_prev[2];
            else
            {
                // Out of a lossy near medium whose cosine was recovered for a real transverse
                // invariant, the direction is the one a transition writes: the Snell ratio with
                // the length N the projection in that medium lost, against the far cosine of the
                // outer pair at the recovered cosine. For a far side of air that is sqrt(p)
                // against sqrt(1 - p), the direction the wave had in air before the wall. The
                // real-angle call above forms the far cosine from eps sin^2 at the traced angle,
                // whose argument is complex for a lossy medium and whose real part does not fall
                // to the external cosine towards grazing: 17.7 deg at 85 deg out of
                // eps 1.50 - 0.40j. Where near and far are one medium, where the recovery
                // declines and where the medium is lossless, nothing here runs and the direction
                // is the one it was.
                double snell_dir = snell;
                std::complex<double> ct2_dir = ct2_far;
                bool dir_real_inv = std::abs(c_n) > 0.0 && ct_slab_real_inv && std::real(c_f) > 0.0;
                if (dir_real_inv)
                {
                    snell_dir *= std::sqrt(std::abs(1.0 - c_n * c_n) + std::real(c_n) * std::real(c_n));
                    ct2_dir = c_f;
                }
                qd_refract(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2],
                           snell_dir, abs_cos_theta_t, ct2_dir, dir_next[0], dir_next[1], dir_next[2]);
                if (dir_real_inv) // a glued exit out of a lossy host: the segment follows
                    RELAUNCH(M_near, M_far, interaction_type);
            }

            if (p_path_dirN)
                p_path_dirN[ii] = (dtype)dir_next[0],
                p_path_dirN[ii + n_rayN] = (dtype)dir_next[1],
                p_path_dirN[ii + 2 * n_rayN] = (dtype)dir_next[2];

            // Total reflection of the outer pair no longer discards the sheet. It used to: the
            // forced-TIR near | far result was written instead, a bare mirror on the reflect pass
            // and the outer pair's tf leak on the transmit pass, which dropped the sheet's loss,
            // its tf, its interface gain and its mass term. Under the acoustic mapping that is
            // every sheet on a rigid wall at every angle. The composite's reflect port needs no
            // forward port in the far medium and is used as it stands; the transmit port is the
            // leak chain qd_slab_resolve forms when told the far side is closed.
            {
                // In-slab propagation angle. Only the refraction cosine is used; the coefficients and
                // the TIR verdict of this near|sheet interface are deliberately discarded for the
                // reason just given.
                std::complex<double> ct2_slab;
                M_near.interact_with(M_slab, interaction_type, theta_t, fGHz, nullptr, nullptr, &ct2_slab, nullptr, nullptr, nullptr);

                // How much of the displaced medium may be taken back out. Air costs nothing to
                // displace and cannot amplify, so it is uncapped. For a real medium the cap is 80% of
                // the host path the tracer can see around the crossing - what it has already travelled
                // plus the gap ahead, when the intersector reported a second face - which leaves at
                // least a tenth of the host on either side of the sheet. A laminate whose layers do not
                // fit inside their host keeps every crossing and simply comes out longer.
                double disp_budget = -1.0;
                const Material &M_disp = disp_far ? M_far : M_near;
                if (!M_disp.same_as(Material()))
                {
                    double host_seen = dist_orig_vbs + ((nH >= 2) ? DIST(F, S) : dist_fbs_dest);
                    disp_budget = 0.8 * host_seen * abs_cos_theta_t;
                }

                // No grazing clamp, matching ray_mesh_interact. A fixed extent-to-thickness ratio
                // used to stand in for the face extent this function cannot see, and it froze the
                // interface past 89.943 deg for every sheet at every thickness. Passing the real
                // extent through instead was the obvious repair and is the one option that had to be
                // rejected: that extent is the longest edge of the hit triangle, so importing it here
                // would have made a production result depend on how finely a wall was meshed, which
                // it currently does not. The sheet is a laterally infinite layer in both functions;
                // the note at the corresponding site in ray_mesh_interact gives the measurements.
                double ct_geo = abs_cos_theta_t;
                double ct_slab = std::abs(std::real(ct2_slab));

                // Incidence cosine in the near medium. ct_geo is the real projection qd_refract left
                // behind, which is a genuine incidence cosine only while the ray arrives from air.
                // Where the near medium is a traced solid the recovery returns the complex value the
                // path actually carried, and produce mode then builds both boundaries, the internal
                // series, the displaced-host propagation and the near-to-far flux conversion from
                // it. Where it declines, produce mode falls back to ct_geo and nothing moves.
                //
                // The recovery names its own entry medium: mtl_prev is the medium this leg was
                // entered through, and behind a sheet it assumes a lossless one.
                std::complex<double> ct_near = EXACT_IN_SLAB_COSINE();

                // Thickness enters only here. Produce mode takes the normal thickness and the two
                // cosines; the geometric-path slot is unused.
                //
                // An ordered stack goes to the characteristic-matrix core instead, because two
                // layers couple through reflections between non-adjacent boundaries and those do not
                // factor into a product of per-layer series. The single-sheet call below is left
                // exactly as it was: the two forms are algebraically equal at one layer and agree to
                // 9.6e-16 wherever no boundary is past its critical angle, but they are not
                // bit-identical, and produce mode's one-layer path is asserted to be.
                // Angle factor of a mass term: the cosine the wave has in air.
                //
                // Reached from air, and from inside a medium that holds the wave undeviated, that
                // is ct_geo. Behind a medium the wave was refracted into it is not: the plate is
                // driven at the trace wavenumber, the term is calibrated on a plate in air, and
                // the cosine in the layer in front would make one wall pay differently from its
                // two sides and differently from the same sheets in air. It is the far cosine of
                // near | air, at the recovered cosine where there is one and at the traced angle
                // where the recovery declines, which is what a closed face hands a traced
                // mass-law wall as its direction. Where the invariant has no direction in air,
                // one or more, the term keeps the cosine the wave has here, as that direction
                // does. The gate is on the media and the state; a leg the state does not know to
                // be bent keeps ct_geo.
                double ct_mass = ct_geo;
                if (mtl_near != 0 && refracted && !(M_near.m > 0.0) && (M_slab.m > 0.0 || (mtl_slab2 != 0 && M_slab2.m > 0.0)))
                {
                    std::complex<double> cos_air;
                    bool tir_air = false;
                    M_near.interact_with(MAT(0), interaction_type, theta_t, fGHz, nullptr, nullptr, &cos_air, nullptr, nullptr, &tir_air,
                                         false, (std::abs(ct_near) > 0.0) ? &ct_near : nullptr, false);
                    if (!tir_air && std::real(cos_air) > 0.0)
                        ct_mass = (std::real(cos_air) > 1.0) ? 1.0 : std::real(cos_air);
                }

                std::complex<double> Tte, Ttm, Rte, Rtm;
                if (mtl_slab2 != 0)
                {
                    QdLayer lay[2];
                    lay[0].mtl = &M_slab, lay[0].thk = thk;
                    lay[1].mtl = &M_slab2, lay[1].thk = thk2;
                    std::complex<double> ci = (std::abs(ct_near) > 0.0)
                                                  ? ct_near
                                                  : std::complex<double>(ct_geo, 0.0);
                    if (!qd_stack_resolve(M_near, lay, 2, M_far, ci, fGHz, is_scalar,
                                          &Tte, &Ttm, &Rte, &Rtm, disp_far, disp_budget, ct_mass, mass_bonded, !refracted))
                        return false; // malformed stack or a singular denominator; the caller decides
                }
                else
                    qd_slab_resolve(M_near, M_slab, M_far, thk, 0.0, ct_slab, ct_geo,
                                    fGHz, is_scalar, 0.0, true, &Tte, &Ttm, &Rte, &Rtm,
                                    disp_far, disp_budget, std::complex<double>(0.0, 0.0), ct_near,
                                    nullptr, tir, !refracted, mass_bonded, nullptr, false, ct_mass);

                // Both ports come out of the one pass. The reflection port is its own Airy sum, not the
                // half-space Fresnel value, so expect real angular and frequency structure in it.
                cTE = refl_pass ? Rte : Tte;
                cTM = refl_pass ? Rtm : Ttm;
            }

            double xprmat[8], pgain;
            qd_polbasis(dir_prev[0], dir_prev[1], dir_prev[2], dir_next[0], dir_next[1], dir_next[2],
                        N[0], N[1], N[2], 1.0, cTE, cTM, is_scalar, xprmat, pgain);
            if (p_xprmatN)
                for (size_t iX = 0; iX < nXPR; ++iX)
                    p_xprmatN[ii * nXPR + iX] = (dtype)xprmat[iX];
            if (p_gainN)
                p_gainN[ii] = (dtype)pgain;
            return true;
        };

        // A thin-sheet crossing is an atomic three-medium event. ray_mesh_interact already summed the
        // internal series, so nothing here may re-apply an interface coefficient through interact_with
        // or charge a medium gain for the sheet material. What this function owes is the part
        // ray_mesh_interact could not know - the bounding media - plus the state bookkeeping around the
        // event.
        //
        // A sheet inside a solid is a three-layer stack, and the existing machinery already models one.
        // The sequence is the same in every case:
        //   1. close out the host leg, whose internal series now reflects off the sheet rather than off
        //      whatever face material the mesh named (SLAB_AIRY_FACTOR with the sheet as mtl_near),
        //   2. re-resolve the sheet against the media the state words report,
        //   3. start a fresh leg with the sheet recorded as the previous medium, so that the next host
        //      leg reflects off it from the other side.
        // Steps 1 and 2 are written in the opposite order below, and have to be: SLAB_AIRY_FACTOR
        // rebuilds xprmat from cTE / cTM, and those only exist once the sheet has been resolved.
        auto SHEET_EVENT = [&]()
        {
            // Ordered stack, sheet_case 4. The two sheets are one atomic crossing resolved by the
            // characteristic-matrix core, in the order the hit code carries. Both sides of the pair
            // are the medium the ray is travelling in, so near == far, the flux conversion is 1, and
            // the stack displaces its whole thickness of that medium. Where the core declines - a
            // lumped column it cannot express - the ray is killed, which is what this case did
            // unconditionally before.
            bool ray_ends = nH == 1 || (nH == 2 && typeH[2]);

            // Bounding media from the tracked state. mtl_current is the near side in every case: the
            // ray is travelling in it when it reaches the sheet. The far side is that same medium for
            // an isolated sheet, the glued solid on entry, and air on exit.
            //
            // Air beyond a glued exit is not read from the state. The previous-medium word names
            // what the host was entered from, or the sheet it was entered through, and either is
            // behind the ray: it is the mirror behind it in the host's series and nothing else. A
            // wall of two solid layers with a coating on its far side has the first layer in that
            // word and air beyond the coating. The medium a host was entered from lies beyond its
            // exit sheet only for a lined body nested in that medium, which is not a use case, and
            // a sheet between two solids is not a geometry the intersector reports; if one occurs
            // the next exit corrects the current medium. Air is also what ray_mesh_interact's own
            // composite assumes for every glued exit.
            int m_near = mtl_current;
            int m_far = (sheet_case == 2) ? sheet_solid : ((sheet_case == 3) ? 0 : mtl_current);
            int m_next = m_far; // medium the ray travels in after the crossing

            // No sheet code in scope carries a deferred buffer. Flag the violation rather than guess:
            // a glued exit still exits into air, and the deferred medium is dropped.
            if (mtl_buffer != 0)
                typeR[4] = true;

            // A sheet whose impedance matches the host is not a boundary, and the host layer must not
            // be cut in two there. Splitting it costs the layer its internal series twice over:
            // fresnel_r returns r = 0 for a matched pair, so the Airy denominator collapses to exactly
            // 1 at this crossing, and again at the far side once the sheet has been written into the
            // previous-medium word. Measured, |S| reads 1.000000 in every such row, worth up to 1 dB.
            //
            // So carry the leg through instead: keep the accumulated distance, leave the state words
            // alone, and let the series resolve once at the far side over the whole path. Only the
            // sheet's own composite is applied here.
            //
            // This stays a special case rather than the general fix. A sheet that merely fails the
            // transparency test still splits the layer, and it is the characteristic-matrix core
            // below that keeps the split honest by resolving the sheet as a layer rather than as a
            // half-space; the merge only avoids doing that work where there is provably no boundary.
            //
            // Restricted to the transmission pass, because a matched sheet reflects nothing and the
            // reflected ray carries no energy either way. Restricted to a single layer as well: an
            // ordered stack is not transparent because its first layer happens to be, and merging on
            // that test would carry the leg through while discarding the second sheet entirely.
            //
            // A mass-law sheet in a mass-law host is not merged. Such a host has no series to
            // protect, and a leg carried through would pay its own term at the far side without
            // the sheet in its plate. Cut at the sheet, the two halves and the sheet are the three
            // bonded bodies of any other sheet inside a wall.
            bool merge_host = !refl_pass && sheet_case != 4 && mtl_current != 0 &&
                              m_next == mtl_current && TRANSPARENT_TO(sheet_mtl, mtl_current) &&
                              !(MAT(mtl_current).m > 0.0 && MAT(sheet_mtl).m > 0.0);

            DIST_ADD(); // Advance the host leg up to the sheet

            // The sheet sits on the side of a glued pair that is not the solid: near for a glued
            // entry, far for a glued exit. Isolated sheets have near == far and are unaffected.
            // sheet_mtl2 is zero except for an ordered stack, where it is the second layer.
            //
            // A sheet reached from inside a mass-law body is bonded to it. The host leg closes at
            // this event and pays its own term below, so the sheet is handed the X0 that leg
            // carries, with whatever the leg itself was entered through, and pays the increment.
            double mass_host = 0.0;
            if (mtl_current != 0)
            {
                Material M_host = MAT(mtl_current);
                if (M_host.m > 0.0)
                    mass_host = MASS_RUN() + ((dist_geo > 0.0015) ? M_host.mass_X0(dist_geo * abs_cos_theta, fGHz) : 0.0);
            }
            bool have_composite = SHEET_RESOLVE(sheet_mtl, m_near, m_far, sheet_case == 3, sheet_mtl2, mass_host);

            if (sheet_case == 4 && !have_composite)
            {
                KILL_RAY(); // no stack physics available, and no legacy result worth keeping
                return;
            }

            // Close out the host medium. Its series reflects off the sheet ahead and off whatever the
            // host was entered through, which is the existing argument order with the sheet index
            // substituted for the face material. The sheet passed as mtl_near is now resolved as a
            // layer of its own thickness rather than as a half-space, so a sheet electrically close
            // to the host no longer collapses the denominator and cost the host its internal series.
            // This is the one site that can name what lies behind that sheet - m_far is still in
            // scope here - so it passes it; a later leg cannot and assumes air. Skipped without a
            // sheet composite, because the factor is applied through cTE / cTM, and for a resolved
            // ray, whose series the reflection behind it already collapsed, as APPLY_TRANSITION
            // skips it at a bare face.
            bool host_summed = false; // the host's bounce series was summed at this event
            if (mtl_current != 0 && !merge_host)
            {
                if (have_composite && !resolved && !followed)
                    host_summed = SLAB_AIRY_FACTOR(mtl_current, sheet_mtl, mtl_prev, m_far);
                SCALE_BY_MEDIUM(mtl_current);
            }

            // Type bits. Bit 6 set together with bit 2 marks a thin-sheet event; that combination is
            // unreachable for any other interaction, because a transparent interface resolves no series
            // and scattering is not implemented. Inside that region bit 1 no longer means VBS - the
            // redo above is always evaluated at theta_t, so a VBS flag would be constant - and carries
            // the glued verdict instead.
            typeR[1] = (sheet_case == 2 || sheet_case == 3); // glued: the two sides differ
            typeR[2] = true;                                 // the internal series is accounted for
            typeR[6] = true;                                 // qualified by bit 2: thin sheet
            typeR[3] = refl_pass ? (mtl_current != 0) : (m_next != 0);

            if (refl_pass) // A reflection stays in the host medium
            {
                DIST_SET();
                if (mtl_current != 0)
                {
                    // Terminate the internal bounce series exactly as a resolved solid does: where
                    // it was summed here. Where it was declined the bounces are the tracer's, and
                    // the reflected ray is followed as it is off a bare face.
                    // The sheet reflected off is behind the ray now. Where both of its sides are
                    // the medium the ray is in, that medium is what backs it.
                    next_current = (short)((mtl_current & 0x3FFF) | (host_summed ? (int)0x8000 : 0) | bent_carry);
                    next_prev = (short)((sheet_mtl & 0x3FFF) | (prev_nonpar ? (int)0x8000 : 0) |
                                        ((m_far == mtl_current) ? (int)0x4000 : 0));
                    next_buffer = (short)(host_summed ? 0 : (int)0x4000); // not summed: followed
                }
                return;
            }

            // Transmission: the leg restarts on the far side of the sheet, unless it was never
            // interrupted. Two cases leave everything as it stands - a merged host layer, and air on
            // both sides, where the accumulators keep running and are dropped at the next medium entry
            // in the ordinary pass-through way.
            if (merge_host || (mtl_current == 0 && m_next == 0))
                return;

            // Clearing the resolved flag starts a new leg. Behind a sheet inside the medium the leg
            // is bent if the one in front was; a glued entry says what it did itself.
            next_current = (short)((m_next & 0x3FFF) |
                                   ((m_next == 0) ? 0 : ((m_next == mtl_current) ? bent_carry : bent_enter)));

            // The sheet becomes the previous medium so the next host leg reflects off it. It is a
            // boundary material here, never a traversed one: mtl_prev is only ever read as the far
            // material of a slab resolve and for the parallelism flag. An exit into air writes it as
            // well: nothing reads the word while the ray is in air, and should the ray turn out to
            // be in a medium after all, the sheet is the boundary behind it.
            //
            // The wedge test is unusable for a glued entry: fbs_sbs_not_parallel compares FBS and SBS,
            // which are the sheet and the solid's near face, and they are co-located by construction.
            // The question it should answer is whether the solid's far face is parallel to its near
            // face, and that face is not in this hit at all. Defaulting to parallel allows the resolve
            // at the far side, which is right for the motivating geometry (a coating on a flat wall)
            // and wrong for a sheet glued to a wedge.
            bool nonpar = (sheet_case == 2) ? false : prev_nonpar;

            // The boundary material behind the ray is the LAST layer it crossed, which differs from
            // sheet_mtl only for an ordered stack.
            //
            // A sheet crossed inside a medium has that medium on both sides, so the leg that
            // starts here has the sheet behind it backed by its own medium, and says so in bit 14.
            int sheet_back = (sheet_mtl2 != 0) ? sheet_mtl2 : sheet_mtl;
            bool inside = m_next != 0 && m_next == mtl_current;
            next_prev = (short)((sheet_back & 0x3FFF) | ((m_next != 0 && nonpar) ? (int)0x8000 : 0) |
                                (inside ? (int)0x4000 : 0));

            // A mass-law sheet crossed into a medium: flag it, so the leg that starts here knows
            // the sheet in the previous-medium word was crossed and not merely reflected off.
            // Crossed inside a mass-law medium, the leg closed here is bonded to the one that
            // starts as well, through the sheet, and its thickness travels as the code, with
            // what that leg was entered through and the first layer of an ordered stack.
            next_buffer = (short)0;
            if (m_next != 0 && MAT(sheet_back).m > 0.0)
            {
                next_buffer = (short)0x8000;
                if (inside && MAT(mtl_current).m > 0.0 && dist_geo > 0.0015)
                    next_buffer = MASS_CODE(MASS_FOLD(mtl_current, dist_geo * abs_cos_theta, (sheet_mtl2 != 0) ? sheet_mtl : 0));
            }

            if (ray_ends && m_next != 0) // Entering or staying in a medium: refracted preload
            {
                double dist_VV;
                if (VBS_INTERSECT(F, dir_next, D, N, &dist_VV))
                    DIST_SET(dist_VV, dist_fbs_dest);
                else
                    DIST_SET(dist_fbs_dest);
            }
            else if (ray_ends) // Leaving into air: the post-exit leg is unrefracted
                DIST_SET(dist_fbs_dest);
            else
                DIST_SET();
        };

        // M2M transition (i-i) and cavity exit (i-o)
        auto APPLY_TRANSITION = [&](bool exit = false,         // Cavity exit flag
                                    int mtl_next = 0,          // Next material: air (exit default)
                                    bool preload_dest = false) // Preloads FBS/VBS to dest distance
        {
            // Update mtl_current based on the object-exit material
            if (mtl_next == 0) // Skip for manual next-material overwrite
            {
                int mtl_exit = GET_EXIT_MATERIAL();
                mtl_current = mtl_exit ? mtl_exit : mtl_current;
            }

            if (!exit && mtl_next == 0) // Auto-select next medium
                mtl_next = typeH == 5 ? M2 : M1;

            DIST_ADD(); // Add orig-fbs/vbs segment lengths

            // Buffer word for the leg that follows. A transparent transition carries the leg on
            // and its word with it; a mass-law solid crossed into another medium leaves the
            // surface mass of the leg closed here and of what it was bonded to; everything else
            // starts clean.
            short buffer_next = (short)0;

            // Transparent transition for same material
            if (SAME(mtl_current, mtl_next)) // Make interface transparent
            {
                if (mass_crossed || followed)
                    buffer_next = buffer_word_in;
                REPLACE_BY_GAIN(1.0, true); // Maintain refracted direction, set transparent flag
                if (preload_dest)           // Add distances: fbs-vbs (geometric) AND vbs-vdest (refracted)
                {
                    double dist_VV;
                    if (VBS_INTERSECT(V, dir_prev, D, N, &dist_VV))
                        DIST_ADD(dist_VV, dist_fbs_dest);
                    else // Fallback, use fbs-dest distance for both
                        DIST_ADD(dist_fbs_dest);
                }
            }
            else // Different materials or exit
            {
                double dist_VV = REPLACE_BY_VBS(mtl_current, mtl_next, (is_scalar ? 4 : 1), (exit ? 3 : 2));

                if (!resolved && !followed) // Resolve multi-bounce series of current layer
                {
                    if (SLAB_AIRY_FACTOR(mtl_current, mtl_next, mtl_prev))
                        typeR[2] = true; // Set slab resolve flag
                }

                SCALE_BY_MEDIUM(mtl_current); // Apply current medium gain and phase

                if (!exit && mtl_next != 0 && MAT(mtl_current).m > 0.0 && dist_geo > 0.0015)
                    buffer_next = MASS_CODE(MASS_FOLD(mtl_current, dist_geo * abs_cos_theta));

                if (preload_dest && exit) // exit into air: post-exit leg is unrefracted, refract == geo
                    DIST_SET(dist_fbs_dest);
                else if (preload_dest) // entered a medium: refracted slant distance differs from geometric
                    DIST_SET(dist_VV, dist_fbs_dest);
                else
                    DIST_SET();

                // These need to be here! Same materials with different ids have to keep the entry material on next_prev for SLAB_AIRY_FACTOR
                // Into a medium, the word also says whether the face was evaluated at the cosine
                // of a wave with a real transverse invariant (prev_real_inv).
                next_prev = (short)(mtl_current | ((face_real_inv && !exit && mtl_next != 0) ? (int)0x4000 : 0));
                // A leg entered out of a mass-law medium is not a bent one: the wave was held there
                next_current = (short)(mtl_next | ((mtl_next != 0 && !exit && !(MAT(mtl_current).m > 0.0)) ? bent_enter : 0)); // Clear resolved flag
            }

            next_buffer = buffer_next; // Clear the deferred medium

            if (!exit) // Set continue-inside flag
                typeR[3] = true;
        };

        if (refl_pass) // Reflection pass, interaction_type {0, 3}
        {
            typeR[7] = true; // Set reflection flag

            int mtl_exit = GET_EXIT_MATERIAL();
            mtl_current = mtl_exit ? mtl_exit : mtl_current;        // Material the ray travels in
            int mtl_next = typeH == 5 ? M2 : (typeH == 7 ? M1 : 0); // Material the ray reflects off

            // Events that trigger a transparent pass-through cannot reflect at the same time
            bool transparent_forward = (nH == 0) || (typeH == 23) || (!typeH[0]) ||
                                       (mtl_current != 0 && SAME(mtl_current, mtl_next)) ||
                                       (mtl_buffer != 0 && SAME(mtl_buffer, M1));

            if (resolved)                 // Internal front reflection already included in internal back reflection
                KILL_RAY();               // This terminates the infinite internal bounce series (typeR 0)
            else if (sheet_case != 0)     // Thin sheet: the composite already carries the reflect port
                SHEET_EVENT();            // (typeR 197, 199, 205, 207)
            else if (transparent_forward) // Ray did not hit anything (= transparent pass-through, cannot reflect)
                KILL_RAY();               // Set typeR 0
            else if (mtl_current == 0)    // Entry front reflection, bare Fresnel copy-through
                DIST_SET();               // Reset distance, flags already set: ok, refl (typeR 129)
            else                          // Internal back reflection of a resolvable parallel slab (typeR 137, 139, 141, 143)
            {
                DIST_ADD();                                                    // Update distances within the current medium
                REPLACE_BY_VBS(mtl_current, mtl_next, (is_scalar ? 3 : 0), 0); // Mirror at VBS (typeR 137, 139)
                next_buffer = (short)0x4000;                                   // Clear buffer; not summed: followed
                if (!followed && SLAB_AIRY_FACTOR(mtl_current, mtl_next, mtl_prev)) // Apply thin-slab (Fabry-Perot) factor
                {
                    next_current = (short)((mtl_current & 0x3FFF) | (int)0x8000 | bent_carry); // keep material, set resolved
                    next_buffer = (short)0;
                    typeR[2] = true; // slab resolve flag
                }
                SCALE_BY_MEDIUM(mtl_current); // Apply medium gain and phase
                DIST_SET();                   // Reset in-medium distance
                typeR[3] = true;              // Set continue-inside flag (typeR 137)
            }
        }
        else // Transmission / refraction pass, interaction_type {1, 2, 4, 5}
        {
            // Note: No separate slab-resolve logic needed.
            // Resolved rays pass like any other ray and APPLY_TRANSITION gates multiple slab resolves.

            // If the ray ends in the next segment, we can already preload its length
            bool ray_ends = nH == 1 || (nH == 2 && typeH[2]);

            if (nH == 0 || !typeH[0]) // No crossing: accumulate the path distance, identity interface
            {
                DIST_ADD();                             // Advance to dest (preload next leg)
                REPLACE_BY_GAIN(1.0, mtl_current != 0); // Pass-through, flags: transparent, ok (typeR 65)
                if (mtl_current != 0)                   // Currently inside a material?
                    typeR[3] = true;                    // Pass-through, flags: transparent, inside, ok (typeR 73)
            }
            else if (sheet_case != 0) // Thin-sheet crossing (typeR 69, 71, 77, 79)
            {
                SHEET_EVENT();
            }
            else if (typeH == 3 || typeH == 15 || typeH == 31) // o-i
            {
                typeR[3] = true;      // Set inside flag (typeR 9)
                if (mtl_current == 0) // Entry from air, keep xprmat and gain unchanged
                {
                    next_current = (short)(M1 | bent_enter);           // Next medium
                    bool nonpar = !ray_ends && fbs_sbs_not_parallel(); // SBS not parallel to FBS?
                    next_prev = (short)(nonpar ? (int)0x8000 : 0);     // air + flag

                    if (typeH == 15 || typeH == 31) // o-ii (degenerate geometry)
                    {                               // Note: For type 15, M1/M2 order is random
                        next_buffer = (short)M2;    // Push M2 into buffer
                        typeR[0] = false;           // Clear OK bit (signal deferred buffer)
                    }

                    if (ray_ends) // Preload next leg
                    {
                        double dist_VV; // Distance from VBS to virtual destination inside the object
                        if (VBS_INTERSECT(F, dir_next, D, N, &dist_VV))
                            DIST_SET(dist_VV, dist_fbs_dest);
                        else // Fallback
                            DIST_SET(dist_fbs_dest);
                    }
                    else
                        DIST_SET(); // entering medium: drop accumulated free-space length, restart at FBS
                }
                else // inside. nested pass-through (o-i on an inside state), continue in cur (typeR 72)
                {
                    APPLY_TRANSITION(false, mtl_current, ray_ends); // Stay in current, clear buffer
                    next_buffer = (short)M1;                        // Store buffer material
                    typeR[0] = false;                               // Clear OK bit (signal deferred buffer)
                }
            }
            else if (typeH == 1 || typeH == 13 || typeH == 29) // i-o
            {
                if (mtl_buffer == 0)                     // i-o: cavity exit
                    APPLY_TRANSITION(true, 0, ray_ends); // Exit to air (typeR 1, 3, 5, 7)
                else if (typeH == 1)                     // virtual i-i, mtl_buffer != 0
                {
                    if (SAME(mtl_buffer, M1))                           // M2 embedded in M1, ignore M2
                        APPLY_TRANSITION(false, mtl_current, ray_ends); // Stay in current, clear buffer
                    else                                                // Transition into buffer material (typeR 9, 11, 13, 15)
                        APPLY_TRANSITION(false, mtl_buffer, ray_ends);  // Clear buffer
                }
                else // typeH 13/29, buf != 0: ii-o, exit to air
                    APPLY_TRANSITION(true, 0, ray_ends);
            }
            else if (typeH == 23) // Corner o-i-o
            {
                if (mtl_current == 0) // Currently in air, ignore corner, stay in air
                {
                    DIST_ADD(dist_orig_fbs); // Accumulate distance orig-fbs
                    REPLACE_BY_GAIN();       // Pass-through (typeR 65)
                }
                else                                     // Inside material, transition to outside
                    APPLY_TRANSITION(true, 0, ray_ends); // Exit to air
            }
            else if (typeH == 21) // Corner i-o-i
            {
                if (mtl_buffer == 0)
                {
                    if (M2 == 0) // illegal
                        KILL_RAY();
                    else // Air gap bounded by M1 / M2, treat as i-i transition
                    {
                        int mtl_exit = GET_EXIT_MATERIAL();              // Resolve M1 = exit material
                        mtl_current = mtl_exit ? mtl_exit : mtl_current; // Should be same as mtl_current
                        APPLY_TRANSITION(false, M2, ray_ends);           // i-i from M1 to M2
                    }
                }
                else if (mtl_current != 0) // current + buffer
                {
                    if (SAME(mtl_buffer, M1))                           // M2 embedded in M1, ignore M2
                        APPLY_TRANSITION(false, mtl_current, ray_ends); // Stay in current, clear buffer
                    else                                                // Transition into buffer material
                        APPLY_TRANSITION(false, mtl_buffer, ray_ends);  // Clear buffer
                }
                else // buf != 0 and cur == 0: terminate
                    KILL_RAY();
            }
            else if (typeH == 5 || typeH == 7) // i-i
            {
                if (mtl_buffer == 0)
                    APPLY_TRANSITION(false, 0, ray_ends);
                else
                {
                    APPLY_TRANSITION(false, mtl_current, ray_ends);        // Stay in current, clear buffer
                    next_buffer = (short)(SAME(mtl_buffer, M1) ? M2 : M1); // Swap buffer
                    typeR[0] = false;                                      // Clear OK bit (signal deferred buffer)
                }
            }
            else // Unmatched
                KILL_RAY();
        }

        // Write the new state words (compact set)
        if (p_prev_outN)
            p_prev_outN[ii] = next_prev;
        if (p_cur_outN)
            p_cur_outN[ii] = next_current;
        if (p_buf_outN)
            p_buf_outN[ii] = next_buffer;
        if (p_acc_geo_outN)
            p_acc_geo_outN[ii] = (dtype)dist_geo;
        if (p_acc_refract_outN)
            p_acc_refract_outN[ii] = (dtype)dist_refract;
        if (p_resolved_typeN)
            p_resolved_typeN[ii] = (uint8_t)typeR;
        if (p_verdict_outN)
            p_verdict_outN[ii] = (uint8_t)series_summed;
    }
}

template void quadriga_lib::ray_state_update(int interaction_type, float center_frequency,
                                             const arma::Mat<float> *orig, const arma::Mat<float> *dest,
                                             const arma::Mat<float> *fbsN, const arma::Mat<float> *sbsN,
                                             const arma::u32_vec *no_interact, const arma::Col<float> *fbs_angleN,
                                             const arma::Mat<float> *normal_vecN, const std::vector<uint8_t> *out_typeN,
                                             const std::unordered_map<std::string, std::vector<float>> *mtl_prop,
                                             const arma::Col<short> *mtl_ind_fbs, const arma::Col<short> *mtl_ind_sbs,
                                             const arma::Col<short> *mtl_ind_prev_in, const arma::Col<short> *mtl_ind_current_in,
                                             const arma::Col<short> *mtl_ind_buffer_in,
                                             const arma::Mat<float> *path_dir_prev,
                                             const arma::Col<float> *acc_dist_geometric, const arma::Col<float> *acc_dist_refracted,
                                             arma::Col<short> *mtl_ind_prev_outN, arma::Col<short> *mtl_ind_current_outN,
                                             arma::Col<short> *mtl_ind_buffer_outN,
                                             arma::Col<float> *gainN, arma::Mat<float> *xprmatN,
                                             arma::Mat<float> *path_dirN,
                                             arma::Col<float> *acc_dist_geometricN, arma::Col<float> *acc_dist_refractedN,
                                             std::vector<uint8_t> *resolved_typeN,
                                             const arma::u32_vec *ray_indN, double eps,
                                             arma::Mat<float> *origN, arma::Mat<float> *destN,
                                             const arma::Mat<float> *tridir, arma::Mat<float> *tridirN,
                                             const std::vector<uint8_t> *verdict_inN, std::vector<uint8_t> *verdict_outN);

template void quadriga_lib::ray_state_update(int interaction_type, double center_frequency,
                                             const arma::Mat<double> *orig, const arma::Mat<double> *dest,
                                             const arma::Mat<double> *fbsN, const arma::Mat<double> *sbsN,
                                             const arma::u32_vec *no_interact, const arma::Col<double> *fbs_angleN,
                                             const arma::Mat<double> *normal_vecN, const std::vector<uint8_t> *out_typeN,
                                             const std::unordered_map<std::string, std::vector<double>> *mtl_prop,
                                             const arma::Col<short> *mtl_ind_fbs, const arma::Col<short> *mtl_ind_sbs,
                                             const arma::Col<short> *mtl_ind_prev_in, const arma::Col<short> *mtl_ind_current_in,
                                             const arma::Col<short> *mtl_ind_buffer_in,
                                             const arma::Mat<double> *path_dir_prev,
                                             const arma::Col<double> *acc_dist_geometric, const arma::Col<double> *acc_dist_refracted,
                                             arma::Col<short> *mtl_ind_prev_outN, arma::Col<short> *mtl_ind_current_outN,
                                             arma::Col<short> *mtl_ind_buffer_outN,
                                             arma::Col<double> *gainN, arma::Mat<double> *xprmatN,
                                             arma::Mat<double> *path_dirN,
                                             arma::Col<double> *acc_dist_geometricN, arma::Col<double> *acc_dist_refractedN,
                                             std::vector<uint8_t> *resolved_typeN,
                                             const arma::u32_vec *ray_indN, double eps,
                                             arma::Mat<double> *origN, arma::Mat<double> *destN,
                                             const arma::Mat<double> *tridir, arma::Mat<double> *tridirN,
                                             const std::vector<uint8_t> *verdict_inN, std::vector<uint8_t> *verdict_outN);