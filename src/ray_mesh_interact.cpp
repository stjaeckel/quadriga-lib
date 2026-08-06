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
            require(att, "att", false);     // penetration loss >= 0
            require(alpha, "alpha", false); // in-medium absorption >= 0
            require(m, "m", false);         // mass-law slope >= 0
            require(resF, "resF", false);   // resonance frequency >= 0 (0 disables)
            require(resQ, "resQ", false);   // resonance Q >= 0 (0 disables)
            require(coiF, "coiF", false);   // coincidence frequency >= 0 (0 disables)
            require(coiQ, "coiQ", false);   // coincidence Q >= 0
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
        double medium_gain(double dist_refract,              // In-medium distance (refracted path)
                           double fGHz = 1.0,                // Frequency
                           double abs_cos_theta_t = 1.0,     // Cosine of incidence angle @ FBS/VBS (refracted path)
                           double dist_geo = 0.0,            // Geometric distance (0.0 = dist_refract)
                           double abs_cos_theta = 0.0) const // Cosine of incidence angle @ orig (only used when dist_geo > 0)
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

            constexpr double mass_min_path = 0.0015;
            if (m > 0.0 && dist_geo > mass_min_path)
            {
                double mass_path = dist_geo * abs_cos_theta * abs_cos_theta;
                double m_dB = m * std::log10((fGHz / fRef) * mass_path);
                if (m_dB > 0.0)
                    loss += m_dB;
            }
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

        // Medium-medium interaction
        // 'this' is the medium the ray travels in (incidence side, medium 1); 'other' is the medium it enters into / reflects off (medium 2).
        // Computes the ITU-R P.2040-1 interface coefficients and returns the interface power gain.
        // For transmission/refraction (1/2/4), the returned gain and cTE/cTM include the entered medium's lumped
        // interface_gain (att + coincidence); reflection (0/3) does not. 0.5*(|cTE|^2 + |cTM|^2) == gain in all cases.
        double interact_with(const Material &other,                         // Material that the path enters into / reflects of
                             int interaction_type,                          // 0 = EM reflect, 1 = EM transmit, 2 = EM refract, 3 = scalar reflect, 4 = scalar transmit, 5 = scalar refract
                             double theta,                                  // Incidence angle
                             double fGHz,                                   // Frequency
                             std::complex<double> *cTE = nullptr,           // Out: E-field coefficient, R for reflection (0/3), T for transmission/refraction (1/2/4)
                             std::complex<double> *cTM = nullptr,           // Out: M-field coefficient
                             std::complex<double> *cos_theta2 = nullptr,    // Out: Refraction cosine (type-2 direction)
                             std::complex<double> *eta1_div_eta2 = nullptr, // Out: eta1/eta2
                             double *Snell_ratio = nullptr,                 // Out: sqrt|eta1*mu1 / eta2*mu2| (Snell ratio, type-2 direction)
                             bool *total_reflection = nullptr,              // Out: Total reflection indicator
                             bool force_tir = false) const                  // Switch to force total internal reflection
        {
            // Interface geometry, shared by every interaction type
            double abs_cos_theta = std::abs(std::cos(theta + 1.570796326794897)); // |OF . N| convention
            abs_cos_theta = (abs_cos_theta > 1.0) ? 1.0 : abs_cos_theta;
            double sin_theta = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta);

            std::complex<double> eta1 = eta(fGHz) + eta_resonance(fGHz); // incidence medium (this)
            std::complex<double> eta2 = other.eta(fGHz) + other.eta_resonance(fGHz);
            std::complex<double> mu1 = mu(fGHz);
            std::complex<double> mu2 = other.mu(fGHz);

            std::complex<double> eta1_d_eta2 = (eta1 * mu1) / (eta2 * mu2);
            double snell = std::sqrt(std::abs(eta1_d_eta2));
            std::complex<double> ct2 = std::sqrt(1.0 - eta1_d_eta2 * sin_theta * sin_theta);
            std::complex<double> z1 = std::sqrt(eta1 / mu1); // admittances sqrt(eps/mu)
            std::complex<double> z2 = std::sqrt(eta2 / mu2);

            bool tir = force_tir || snell * sin_theta >= 1.0;

            std::complex<double> coeff_TE, coeff_TM;
            double gain;

            if (interaction_type < 3) // EM: types 0 (reflection), 1 (transmission), 2 (refraction)
            {
                // Fresnel reflection (TE/TM), ITU-R P.2040-1 eq. (31); under TIR the interface is a perfect mirror
                std::complex<double> R_TE = tir ? std::complex<double>(1.0, 0.0) : (z1 * abs_cos_theta - z2 * ct2) / (z1 * abs_cos_theta + z2 * ct2);
                std::complex<double> R_TM = tir ? std::complex<double>(1.0, 0.0) : (z2 * abs_cos_theta - z1 * ct2) / (z2 * abs_cos_theta + z1 * ct2);
                double reflectance = tir ? 1.0 : 0.5 * (std::norm(R_TE) + std::norm(R_TM));

                // ISSUE: reflectance is constructed from the average R_TE, R_TM, but the two should probably be treated
                // independently. There is a companion fix in SLAB_AIRY_FACTOR that corrects the per-port transmittance.
                // Touching this here will need a corresponding fix as well in SLAB_AIRY_FACTOR to avoid double-correction.

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
                else // EM transmission (1) or refraction (2)
                {
                    // Fresnel transmission, ITU-R P.2040-1 eq. (32)
                    std::complex<double> T_TE = tir ? std::complex<double>(0.0, 0.0) : (2.0 * z1 * abs_cos_theta) / (z1 * abs_cos_theta + z2 * ct2);
                    std::complex<double> T_TM = tir ? std::complex<double>(0.0, 0.0) : (2.0 * z1 * abs_cos_theta) / (z2 * abs_cos_theta + z1 * ct2);
                    double refraction_gain = tir ? 0.0 : 0.5 * (std::norm(T_TE) + std::norm(T_TM));

                    double R_eff = apply_tf_pair(other, reflectance, fGHz);

                    if (interaction_type == 1) // undeviated transmission (1): energy-conserving forward beam = 1 - R_eff
                    {
                        gain = 1.0 - R_eff;
                        if (refraction_gain > 0.0) // rescale the Fresnel T coefficients to carry the forward gain
                        {
                            double s = std::sqrt(gain / refraction_gain);
                            coeff_TE = T_TE * s, coeff_TM = T_TM * s;
                        }
                        else // no Fresnel forward port (TIR, or grazing) -> flat coefficients
                            coeff_TE = coeff_TM = std::complex<double>(std::sqrt(gain), 0.0);
                    }
                    else // refraction (2)
                    {
                        double tf_scale = (reflectance < 1.0) ? (1.0 - R_eff) / (1.0 - reflectance) : 0.0;
                        gain = tir ? 1.0 - R_eff : refraction_gain * tf_scale;
                        coeff_TE = tir ? std::complex<double>(std::sqrt(gain), 0.0) : T_TE * std::sqrt(tf_scale);
                        coeff_TM = tir ? std::complex<double>(std::sqrt(gain), 0.0) : T_TM * std::sqrt(tf_scale);
                    }
                }
            }
            else // Scalar (acoustic): types 3 (reflection), 4 (transmission), 5 (refraction)
            {
                std::complex<double> R = tir ? std::complex<double>(1.0, 0.0) : (z1 * abs_cos_theta - z2 * ct2) / (z1 * abs_cos_theta + z2 * ct2);
                double reflectance = tir ? 1.0 : std::norm(R);
                double R_eff = apply_tf_pair(other, reflectance, fGHz);
                std::complex<double> T = 1.0 + R; // pressure transmission coefficient

                if (interaction_type == 3) // Scalar reflection: tf-adjusted reflectance
                {
                    gain = R_eff;
                    coeff_TE = std::polar(std::sqrt(gain), std::arg(R));
                }
                else if (interaction_type == 4 || tir) // Undeviated transmission (4), or refraction (5) collapsed under TIR
                {
                    gain = 1.0 - R_eff;
                    coeff_TE = std::polar(std::sqrt(gain), std::arg(T));
                }
                else // interaction_type == 5, non-TIR: scalar refraction, field power |T|^2 scaled by the tf shift
                {
                    double tf_scale = (reflectance < 1.0) ? (1.0 - R_eff) / (1.0 - reflectance) : 0.0;
                    gain = std::norm(T) * tf_scale;
                    coeff_TE = T * std::sqrt(tf_scale);
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
  - An acoustic mass-law term `m·log10((f/fRef)·dist)` dB, added once (not per meter) when `m > 0`,
    `dist` exceeds ~1.5 mm, and the term is positive; `m` is the mass-law slope in dB/decade.
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
- Types 3–5 (scalar) use a single TE-only coefficient (acoustic simulation with impedance-mapped
  materials, `ε` derived from `Z`); total internal reflection is handled as in the EM path
  (`snell·sinθ ≥ 1` ⇒ unit reflection, refraction collapses to undeviated transmission).
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
- **`thicknessN`** — Material thickness (FBS-to-SBS distance); `[n_rayN]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (∞ if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code, bit-encoded (`qd::bits<uint8_t>`); `[n_rayN]`<br><br>
   |  Bit | Meaning                                                                 |
   | :--: | ----------------------------------------------------------------------- |
   |   0  | OK flag (0 = no valid interaction / undefined)                          |
   |   1  | Front-side flag (1 = front: o→i or M2 hit first; 0 = back: i→o or M1)   |
   |   2  | Co-located FBS/SBS flag (1 = single point, required for media-to-media) |
   |   3  | Same-direction flag (FBS and SBS normals point the same way)            |
   |   4  | Corner-hit flag (FBS/SBS faces not parallel)                            |
   |   5  | Total-reflection flag (also set when a transmission factor forced it)   |
   Reachable composite values (add 32 for the total-reflection variant):<br><br>
   | Code  |  TIR  | Description                                         |
   | :---: | :---: | --------------------------------------------------- |
   |   0   |   —   | No hit                                              |
   |   1   |  33   | Single hit, inside→outside (exit)                   |
   |   3   |  35   | Single hit, outside→inside (entry)                  |
   |   5   |  37   | Media-to-media, M1 (current, back) hit first        |
   |   7   |  39   | Media-to-media, M2 (next, front) hit first          |
   |  13   |  45   | Overlapping faces, inside-inside→outside            |
   |  15   |  47   | Overlapping faces, outside→inside-inside            |
   |  21   |  53   | Corner hit, inside→outside→inside                   |
   |  23   |  55   | Corner hit, outside→inside→outside                  |
   |  29   |  61   | Corner hit, inside-inside→outside                   |
   |  31   |  63   | Corner hit, outside→inside-inside                   |
- **`path_dirN`** — Refraction-correct path direction: mirror for types 0/3, Snell direction for types 1/2/4/5; `[n_rayN, 3]`.
  For undeviated transmission (types 1/4) this is the *refracted* direction, which differs from the geometric continuation
  (along the incoming ray) used for `origN`/`destN`; it lets downstream code recover the true transmission angle.
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
    MaterialCols<dtype> cols = (mtl_prop) ? MaterialCols<dtype>(*mtl_prop) : MaterialCols<dtype>();
    if (p_mtl_ind && mtl_ind->max() > cols.n_mtl)
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
        if (colocated_faces)
        {
            const double lim = 1.0e-4;
            if (std::abs(Nx + Mx) < lim && std::abs(Ny + My) < lim && std::abs(Nz + Mz) < lim)
                material_to_material = true; // Opposing normal vectors, Material to material transition
            else if (std::abs(Nx - Mx) < lim && std::abs(Ny - My) < lim && std::abs(Nz - Mz) < lim)
                out_flags[3] = true; // Equal normal vectors = overlapping or duplicate faces
            else                     // FBS/SBS faces not parallel
            {
                out_flags[4] = 1; // Corner hit flag
                if ((out_flags[1] && thetaS >= 0.0) ||
                    (!out_flags[1] && thetaS < 0.0))
                    out_flags[3] = true; // Same direction flag
            }
        }

        // Flip normal vector in case of back side illumination
        if (thetaF < 0.0)
            Nx = -Nx, Ny = -Ny, Nz = -Nz,
            cos_theta = ODx * Nx + ODy * Ny + ODz * Nz,
            cos_theta = (cos_theta < -1.0) ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta);

        // Limit value to 0 ... 1 for calculating reflection and transmission coefficients
        double abs_cos_theta = std::abs(cos_theta);

        // Incidence-side (medium 1) and entered/reflected-off (medium 2) materials, default air.
        // 1-based: Material(cols, 0) -> air. M2 always carries the FBS-face material whose
        // interface_gain is the transition gain (front: iMF; back: air, or iMS for M2M).
        Material M1, M2;
        if (thetaF >= 0.0) // front hit: entered material = FBS face (iMF)
        {
            M2 = Material(cols, iMF);
            if (material_to_material) // SBS (front) hit first
                M1 = Material(cols, iMS);
        }
        else // back hit: incidence material = FBS face (iMF)
        {
            M1 = Material(cols, iMF);
            if (material_to_material) // FBS (back) hit first, entered = SBS
                M2 = Material(cols, iMS);
        }

        // Interface evaluation (single source): coefficients + geometry quantities.
        std::complex<double> cTE, cTM, cos_theta2, eta1_div_eta2;
        double eta; // Snell ratio sqrt|eta1*mu1 / eta2*mu2|
        bool total_reflection;
        M1.interact_with(M2, interaction_type, thetaF, fGHz, &cTE, &cTM, &cos_theta2, &eta1_div_eta2, &eta, &total_reflection);
        bool tir_central = total_reflection; // pre-ray-tube TIR state

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

            // Whole-tube TIR: pass through (undeviated) if the spine OR any edge is past critical
            if (geometry_type == 2)
                for (int i = 0; i <= 3 && !total_reflection; ++i)
                    if (eta * sin_thetaV[i] >= 1.0)
                        total_reflection = true;
        }

        if (geometry_type == 0) // Reflection, normalized by default
            qd_reflect(ODx, ODy, ODz, Nx, Ny, Nz, FDx, FDy, FDz);
        else if (geometry_type == 1 || total_reflection) // Transmission without refraction
            FDx = ODx, FDy = ODy, FDz = ODz;             // New path direction = same as incoming ray, already normalized
        else                                             // Refraction
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
                    if (geometry_type == 2 && !total_reflection) // Refraction (skipped when the whole tube passes through under TIR)
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

        // Re-evaluate coefficients if the ray tube introduced TIR (type-2 tube vertices)
        if (total_reflection != tir_central)
            M1.interact_with(M2, interaction_type, thetaF, fGHz, &cTE, &cTM, nullptr, nullptr, nullptr, nullptr, true);

        // Read the output ray index
        if (i_rayN >= n_rayN) // Just to be sure to avoid any segfaults
            throw std::invalid_argument("Something went wrong. This should never be reached!");

        // Write ray_indN (inverse of the compaction map, 0-based input ray index)
        if (p_ray_indN)
            p_ray_indN[i_rayN] = (unsigned)iRx;

        // Relaunch offset: a few float ULP at the FBS coordinate magnitude, so the nudge survives the
        // dtype(origN) store and the relaunched ray starts off the face without re-hitting it
        double scale = std::max({std::abs(Fx), std::abs(Fy), std::abs(Fz), 1.0});
        double relaunch_offset = 8.0 * scale * 1.1920929e-7; // ~8 float ULP at this magnitude (2^-23)
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
            if (geometry_type == 1 && !tir_central) // spine Snell; under spine TIR there is no Snell direction
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
  current medium, the previous medium, and a one-slot next-transition buffer (bit-masked: `mat = w &
  0x7FFF`, `flag = w & 0x8000`).
- Implements the inside/outside state machine and overlays a closed-form thin-slab factor `S` (the Airy
  sum) so a single coefficient captures the full internal multiple-reflection series of a parallel slab
  thin enough to matter, instead of relying on the tracer to follow every internal bounce.
- Called twice per interaction by the ray tracer: once for the reflection pass (`interaction_type` 0
  or 3) and once for the transmission/refraction pass (`interaction_type` 1, 2, 4, 5). With `S`
  suppressed (the survival gate re-emits) the transmission/refraction path reproduces [[calc_diffraction_gain]]

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
    const arma::Mat<dtype> *acc_dist_in = nullptr,
    arma::Col<short> *mtl_ind_prev_outN = nullptr,
    arma::Col<short> *mtl_ind_current_outN = nullptr,
    arma::Col<short> *mtl_ind_buffer_outN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,
    arma::Mat<dtype> *acc_dist_outN = nullptr,
    std::vector<uint8_t> *resolved_typeN = nullptr,
    const arma::u32_vec *ray_indN = nullptr,
    double eps = 0.15);
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
- **`out_typeN`** — Interaction type code from [[ray_mesh_interact]], compact set; `[n_rayN]`
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]])
- **`mtl_ind_fbsN`**, **`mtl_ind_sbsN`** — Material indices M1 / M2 of the FBS / SBS faces, compact set; `[n_rayN]` (0 = air)
- **`mtl_ind_prev_in`**, **`mtl_ind_current_in`**, **`mtl_ind_buffer_in`** — State words, full ray set; `[n_ray]`,
  read at `g`, never written. NULL reads as state `0` (outside, no flags).
- **`path_dir_prev`** — Physical ray direction entering this segment, full ray set; `[n_ray, 3]`
- **`acc_dist_in`** — Accumulated in-layer distance carried into this call, full ray set; `[n_ray, 2]`; col 1 = refracted distance; col 2 = geometric distance
- **`ray_indN`** — Compact-to-full ray index map; `[n_rayN]` to `[n_ray]`; NULL = identity (`n_ray == n_rayN`)
- **`eps`** *(optional)* — Thin-slab (Fabry-Pérot) resolve threshold on the round-trip in-slab
  amplitude `ρ` (`ρ ∈ [0, 1]`): the closed-form Airy factor is applied when `ρ ≥ eps` and the series
  is re-emitted to the tracer when `ρ < eps` (weak / fast-decaying slabs). `eps = 0` always resolves
  (for callers that cannot re-emit, e.g. [[calc_diffraction_gain]]); `eps ≥ 1` always re-emits
  (resolution disabled). A near-pole `S` or a known non-parallel slab re-emits regardless of `eps`.
  Default `0.15`.

## Outputs:
- **`mtl_ind_prev_outN`**, **`mtl_ind_current_outN`**, **`mtl_ind_buffer_outN`** — Updated state words,
  compact set; `[n_rayN]`. NULL skips the write. Passing all six state args NULL disables tracking —
  each interaction is corrected on its own (entry loss, TR kill, single-hit air-gap `S`); cross-interaction slab `S` and
  reflection-bounce `S` need the tracked medium.
- **`gainN`** *(in/out)* — Per-interaction linear power gain, updated in place; `[n_rayN]`. Kept consistent
  with `xprmatN` at every write. A killed ray (`resolved_typeN == 0`) yields `gainN = 0` together with
  an all-zero `xprmatN`.
- **`xprmatN`** *(in/out)* — Polarization transfer matrix, updated in place; `[8, n_rayN]` for EM mode,
  `[2, n_rayN]` for scalar mode. Same layout and basis convention as in [[ray_mesh_interact]], but with the
  medium closed out: on return it additionally contains the in-medium attenuation and excess phase of the
  traversed segment and, when the resolve bit of `resolved_typeN` is set, the closed-form thin-slab (Airy)
  factor `S` covering the full internal multiple-reflection series. FSPL / spreading loss remains excluded,
  so the matrix is the complete per-interaction Jones factor and can be left-multiplied directly into a
  per-path product. When the series is re-emitted instead (`ρ < eps`, resolve bit clear), the matrix carries
  only the first-pass coefficient and the remaining bounces arrive as separate interactions.
- **`path_dirN`** *(in/out)* — Continuation direction, corrected in place by the VBS construction, compact set; `[n_rayN, 3]`
- **`acc_dist_outN`** — Accumulated VBS distance leaving this call, compact set; `[n_rayN, 2]`
- **`resolved_typeN`** *(optional)* — Resolved interaction-type code, bit-encoded (`qd::bits<uint8_t>`),
  compact set; `[n_rayN]`. 0 = ray killed. NULL skips the write.<br><br>
   | Bit  | Flag        | Meaning                                                                        |
   | :--: | :---------: | ------------------------------------------------------------------------------ |
   |   0  | ok          | OK flag (0 = a deferred degenerate-resolve buffer is pending)                  |
   |   1  | vbs         | VBS correction (gain/xprmat corrected at the VBS instead of FBS/SBS)           |
   |   2  | resolve     | Slab-resolve flag (an internal multi-bounce series was resolved analytically)  |
   |   3  | inside      | Inside-object flag (1 = ray continues inside, 0 = continues outside)           |
   |   4  | fix         | Fix flag (resolved-false-outside, or entry/exit material mismatch)             |
   |   5  | tir         | Total-reflection flag (also set when a transmission factor forced reflection)  |
   |   6  | trans       | Transmission: transparent-interface flag; reflection: scatter flag             |
   |   7  | refl        | Reflection flag (0 = transmission/refraction, 1 = reflection/scattering)       |
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
   Reachable composite values for reflection:<br><br>
   |  Dec | Hex  |  FIX |   TIR   | Flags set                       | Meaning                                                      |
   | :--: | :--: | :--: | :-----: | ------------------------------- |------------------------------------------------------------- |
   |  129 | 0x81 |    — |   161   | refl, ok                        | eager front reflection (R0), outside (FBS==VBS)              |
   |  137 | 0x89 |  153 | 169,185 | refl, inside, ok                | internal back-reflection (incoming refr. 2/5, FBS==VBS)      |
   |  139 | 0x8B |  155 | 171,187 | refl, inside, vbs, ok           | internal back-reflection (incoming undev. 1/4, VBS)          |
   |  141 | 0x8D |  157 | 173,189 | refl, inside, resolve, ok       | internal back-reflection + slab series (incoming refr. 2/5)  |
   |  143 | 0x8F |  159 | 175,191 | refl, inside, resolve, vbs, ok  | internal back-reflection + slab series (incoming undev. 1/4) |
   |  192+| 0xC0+|    — |    —    | refl, trans, ...                | reserved: scattering not implemented                         |


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
                                    const arma::Mat<dtype> *acc_dist_in,
                                    arma::Col<short> *mtl_ind_prev_outN,
                                    arma::Col<short> *mtl_ind_current_outN,
                                    arma::Col<short> *mtl_ind_buffer_outN,
                                    arma::Col<dtype> *gainN,
                                    arma::Mat<dtype> *xprmatN,
                                    arma::Mat<dtype> *path_dirN,
                                    arma::Mat<dtype> *acc_dist_outN,
                                    std::vector<uint8_t> *resolved_typeN,
                                    const arma::u32_vec *ray_indN,
                                    double eps)
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

    auto check_state_words = [&](const arma::Col<short> *v, const char *name)
    {
        if (v == nullptr || v->n_elem == 0)
            return;
        for (const short *p = v->memptr(), *pe = p + v->n_elem; p < pe; ++p)
            if (arma::uword(*p & (short)0x7FFF) > cols.n_mtl)
                throw std::invalid_argument(std::string("Values in '") + name + "' exceed the number of materials in 'mtl_prop'.");
    };
    check_state_words(mtl_ind_prev_in, "mtl_ind_prev_in");
    check_state_words(mtl_ind_current_in, "mtl_ind_current_in");
    check_state_words(mtl_ind_buffer_in, "mtl_ind_buffer_in");

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

    if (acc_dist_in && acc_dist_in->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'acc_dist_in' must match the number of rays in 'orig'.");
    const dtype *p_acc_dist_in = acc_dist_in ? acc_dist_in->memptr() : nullptr;

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

    if (acc_dist_outN && (acc_dist_outN->n_rows != n_rayN || acc_dist_outN->n_cols != 2))
        acc_dist_outN->set_size(n_rayN, 2);
    dtype *p_acc_dist_outN = acc_dist_outN ? acc_dist_outN->memptr() : nullptr;

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

        // Extract current material indices and flags
        int mtl_current = next_current & 0x7FFF;
        int mtl_prev = next_prev & 0x7FFF;
        int mtl_buffer = next_buffer & 0x7FFF;
        bool resolved = (next_current & 0x8000) != 0;
        bool prev_nonpar = (next_prev & 0x8000) != 0;

        // Compact-set reads at i
        unsigned nH = p_no_interact[i_ray];                // No interactions
        auto typeH = (qd::bits<uint8_t>)p_out_typeN[ii];   // Hit type code from ray_mesh_interact
        int M1 = (int)(p_mtl_ind_fbs[ii] & (short)0x7FFF); // FBS material
        int M2 = (int)(p_mtl_ind_sbs[ii] & (short)0x7FFF); // SBS material
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

        // Accumulated in-medium distance
        double dist_refract = p_acc_dist_in ? (double)p_acc_dist_in[i_ray] : 0.0;
        double dist_geo = p_acc_dist_in ? (double)p_acc_dist_in[i_ray + n_ray] : 0.0;
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

        // VBS-corrected coefficients
        std::complex<double> cTE(0.0, 0.0), cTM(0.0, 0.0);

        // Replace the FBS-relative interface result ray_mesh_interact wrote with the VBS-equivalent at theta_t
        // - interface coefficients (interface_gain folded inside interact_with), incoming basis from dir_in,
        //   outgoing basis from the corrected continuation. In-medium magnitude/phase stay deferred to close_med.
        // - geom: 0 = reflect at VBS, 2 = Snell refract at VBS (undeviated under TIR), 3 = exit (keep the traced origN->destN direction).
        auto REPLACE_BY_VBS = [&](int Ma, int Mb, int itype, int geom) -> double
        {
            // A resolved path will not call SLAB_AIRY_FACTOR, so we don't need the coefficients.
            if (resolved)
                return dist_fbs_dest;

            // Compute coefficients (cTE and cTM are required by SLAB_AIRY_FACTOR, so we always compute them)
            // only cTE/cTM are needed past the fast return; the geometry outs are used only in the VBS branch
            std::complex<double> cos_t2, e1de2;
            double xprmat[8], pgain;
            double snell = 1.0;
            bool tir = false;
            MAT(Ma).interact_with(MAT(Mb), itype, theta_t, fGHz, &cTE, &cTM, &cos_t2, &e1de2, &snell, &tir);
            typeR[5] = tir; // TIR re-evaluated at the corrected VBS angle (overrides the geometric-angle verdict)

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

            if (!use_vbs) // Fast return
                return dist_fbs_dest;

            typeR[1] = true; // Set VBS correction flag

            // Update the outgoing direction based on the VBS geometry (using the refracted incoming direction)
            if (geom == 0) // Reflect at VBS
                qd_reflect(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], dir_next[0], dir_next[1], dir_next[2]);
            else if (geom == 2 && !tir) // Snell refract at VBS (undeviated under TIR)
                qd_refract(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], snell, abs_cos_theta_t, cos_t2, dir_next[0], dir_next[1], dir_next[2]);

            if (p_path_dirN && geom != 3) // write the corrected continuation; an exit keeps the traced direction
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

        // Apply medium gain and phase
        auto SCALE_BY_MEDIUM = [&](int m)
        {
            Material mtl = MAT(m);
            double g = mtl.medium_gain(dist_refract, fGHz, abs_cos_theta_t, dist_geo, abs_cos_theta);
            double n_re = std::real(std::sqrt(mtl.eta(fGHz) * mtl.mu(fGHz)));
            double k0 = 2.0 * 3.14159265358979323846 * fGHz * 1e9 / 299792458.0;

            // Walk-off (cos^2) only for the undeviated tracer, where it substitutes for the untraced
            // lateral shift. Genuine refraction traced the bent path, so cos2 = 1. The free-space
            // reference sits on the same thickness axis, so dist_geo carries the geometric-incidence cos^2.
            bool walk_off = use_vbs || refl_pass || resolved;
            double cos2 = walk_off ? abs_cos_theta_t * abs_cos_theta_t : 1.0; // medium side, theta_t
            double cos2_geo = walk_off ? abs_cos_theta * abs_cos_theta : 1.0; // free-space side, theta_i
            double excess_phase = k0 * (n_re * dist_refract * cos2 - dist_geo * cos2_geo);

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

        // Analytic thin-slab (Fabry-Perot) resolution
        // Folds the per-polarization Airy factor back into xprmat so TE and TM carry their own slab
        // retardation (and the resulting depolarization). The total power gain is held at the TE/TM-averaged
        // value - identical to the magnitude-only version - by scaling both channels with a common magnitude
        // and only the per-channel phase; the per-channel magnitude of S is not split (that would reweight
        // the gain). Scalar mode, reflection, no-VBS and gain-only cases keep the previous single-factor behavior.
        // Returns true when the series resolved (caller sets any extra state, e.g. the reflected slab flag).
        auto SLAB_AIRY_FACTOR = [&](int mtl_slab, int mtl_near, int mtl_far) -> bool
        {
            if (eps > 0.0 && prev_nonpar) // known wedge/edge -> re-emit (eps = 0.0 always resolves)
                return false;

            Material mat_slab = MAT(mtl_slab), mat_near = MAT(mtl_near), mat_far = MAT(mtl_far);

            // Mass-law materials carry a lumped transmission surrogate: eta/mu are calibrated for
            // surface impedance only, so the in-slab index (and hence the phase of phi) is not
            // physical and the internal series must not be resummed. Re-emit; the tracer follows
            // the bounces and each traversal pays medium_gain once.
            if (mat_slab.m > 0.0)
                return false;

            const double c0 = 299792458.0;
            const double omega = 2.0 * 3.14159265358979323846 * fGHz * 1e9;

            bool slab_is_air = mat_slab.same_as(Material());
            std::complex<double> eta_s_if = mat_slab.eta(fGHz) + mat_slab.eta_resonance(fGHz);
            std::complex<double> mu_s = mat_slab.mu(fGHz);
            double n_cur = std::real(std::sqrt(mat_slab.eta(fGHz) * mu_s)); // resonance-excluded slab index
            double sin2 = 1.0 - abs_cos_theta_t * abs_cos_theta_t;          // incidence sine (fbs_angleN convention)

            // Fresnel TE/TM amplitude reflection at slab|adjacent from the slab side, tf folded into |r|
            auto fresnel_r = [&](const Material &adj,
                                 std::complex<double> &r_te, std::complex<double> &r_tm,
                                 double &R_te, double &R_tm)
            {
                std::complex<double> eta_a_if = adj.eta(fGHz) + adj.eta_resonance(fGHz);
                std::complex<double> mu_a = adj.mu(fGHz);
                std::complex<double> z1 = std::sqrt(eta_s_if / mu_s);
                std::complex<double> z2 = std::sqrt(eta_a_if / mu_a);
                std::complex<double> rat = (eta_s_if * mu_s) / (eta_a_if * mu_a);
                std::complex<double> cos_t2 = std::sqrt(1.0 - rat * sin2);
                std::complex<double> te = (z1 * abs_cos_theta_t - z2 * cos_t2) / (z1 * abs_cos_theta_t + z2 * cos_t2);
                std::complex<double> tm = is_scalar ? te : (z2 * abs_cos_theta_t - z1 * cos_t2) / (z2 * abs_cos_theta_t + z1 * cos_t2);
                const Material &owner = slab_is_air ? adj : mat_slab; // tf of the solid face
                R_te = owner.apply_tf(std::norm(te), fGHz);
                R_tm = owner.apply_tf(std::norm(tm), fGHz);
                r_te = std::polar(std::sqrt(R_te), std::arg(te));
                r_tm = std::polar(std::sqrt(R_tm), std::arg(tm));
            };

            std::complex<double> rn_te, rn_tm, rf_te, rf_tm;
            double Rn_te = 0.0, Rn_tm = 0.0, Rf_te = 0.0, Rf_tm = 0.0;
            fresnel_r(mat_near, rn_te, rn_tm, Rn_te, Rn_tm);
            fresnel_r(mat_far, rf_te, rf_tm, Rf_te, Rf_tm);

            // One-way in-slab propag. phi (polarization-independent): magnitude from medium_gain, phase from n_cur
            double gL = mat_slab.medium_gain(dist_refract, fGHz, abs_cos_theta_t, dist_geo, abs_cos_theta);
            double abs_phi = std::sqrt((gL < 0.0) ? 0.0 : gL);
            double arg_phi = -(omega / c0) * n_cur * dist_refract * abs_cos_theta_t * abs_cos_theta_t;
            std::complex<double> phi2 = std::polar(abs_phi * abs_phi, 2.0 * arg_phi);

            std::complex<double> denom_te = std::complex<double>(1.0, 0.0) - rn_te * rf_te * phi2;
            std::complex<double> denom_tm = std::complex<double>(1.0, 0.0) - rn_tm * rf_tm * phi2;

            // Survival gate (stronger polarization) + near-pole clamp (either) -> re-emit
            double g2L = mat_slab.medium_gain(2.0 * dist_refract, fGHz, abs_cos_theta_t, 2.0 * dist_geo, abs_cos_theta);
            g2L = (g2L < 0.0) ? 0.0 : g2L;
            double rr_te = (Rn_te * Rf_te < 0.0) ? 0.0 : Rn_te * Rf_te;
            double rr_tm = (Rn_tm * Rf_tm < 0.0) ? 0.0 : Rn_tm * Rf_tm;
            double rho = std::sqrt(((rr_te > rr_tm) ? rr_te : rr_tm) * g2L);
            if (rho < eps || std::abs(denom_te) < 1.0e-2 || std::abs(denom_tm) < 1.0e-2)
                return false;

            std::complex<double> S_te = std::complex<double>(1.0, 0.0) / denom_te;
            std::complex<double> S_tm = std::complex<double>(1.0, 0.0) / denom_tm;

            if (is_scalar) // single coefficient: keep the complex factor (phase carried into xprmat[0..1])
            {
                SCALE_RAY(std::real(S_te), std::imag(S_te));
                return true;
            }

            // Per-channel single-pass transmittance (two interfaces) and the averaged value the ports carry
            double Tn_avg = 1.0 - 0.5 * (Rn_te + Rn_tm);
            double Tf_avg = 1.0 - 0.5 * (Rf_te + Rf_tm);
            double single_pass = Tn_avg * Tf_avg;          // = |slab_cTE|^2 = |slab_cTM|^2 (averaged ports)
            double Tsp_te = (1.0 - Rn_te) * (1.0 - Rf_te); // true TE single-pass transmittance
            double Tsp_tm = (1.0 - Rn_tm) * (1.0 - Rf_tm); // true TM single-pass transmittance

            // Full complex Airy factor per channel, plus port correction from averaged -> per-channel magnitude.
            // |fac|^2 * |port|^2 = Tsp * |S|^2 = the exact per-channel slab transmittance (energy-correct, coherent).
            double corr = (single_pass > 1.0e-12) ? 1.0 / single_pass : 0.0;
            std::complex<double> fac_te = S_te * std::sqrt(Tsp_te * corr);
            std::complex<double> fac_tm = S_tm * std::sqrt(Tsp_tm * corr);

            // ISSUE: This corrects the average per-channel magnitude used in Material.interact_with, which may need
            // to be fixed. So IF of get fixed, this here needs updating too to avoid double-correction.

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

            // Transparent transition for same material
            if (SAME(mtl_current, mtl_next)) // Make interface transparent
            {
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

                if (!resolved) // Resolve multi-bounce series of current layer
                {
                    if (SLAB_AIRY_FACTOR(mtl_current, mtl_next, mtl_prev))
                        typeR[2] = true; // Set slab resolve flag
                }

                SCALE_BY_MEDIUM(mtl_current); // Apply current medium gain and phase

                if (preload_dest && exit) // exit into air: post-exit leg is unrefracted, refract == geo
                    DIST_SET(dist_fbs_dest);
                else if (preload_dest) // entered a medium: refracted slant distance differs from geometric
                    DIST_SET(dist_VV, dist_fbs_dest);
                else
                    DIST_SET();

                // These need to be here! Same materials with different ids have to keep the entry material on next_prev for SLAB_AIRY_FACTOR
                next_prev = (short)mtl_current;
                next_current = (short)mtl_next; // Clear resolved flag
            }

            next_buffer = (short)0; // Clear buffer

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
            else if (transparent_forward) // Ray did not hit anything (= transparent pass-through, cannot reflect)
                KILL_RAY();               // Set typeR 0
            else if (mtl_current == 0)    // Entry front reflection, bare Fresnel copy-through
                DIST_SET();               // Reset distance, flags already set: ok, refl (typeR 129)
            else                          // Internal back reflection of a resolvable parallel slab (typeR 137, 139, 141, 143)
            {
                DIST_ADD();                                                    // Update distances within the current medium
                REPLACE_BY_VBS(mtl_current, mtl_next, (is_scalar ? 3 : 0), 0); // Mirror at VBS (typeR 137, 139)
                if (SLAB_AIRY_FACTOR(mtl_current, mtl_next, mtl_prev))         // Apply thin-slab (Fabry-Perot) factor
                {
                    next_current = (short)((mtl_current & 0x7FFF) | (int)0x8000); // keep material, set resolved
                    typeR[2] = true;                                              // slab resolve flag
                }
                SCALE_BY_MEDIUM(mtl_current); // Apply medium gain and phase
                DIST_SET();                   // Reset in-medium distance
                next_buffer = (short)0;       // Clear buffer (match APPLY_TRANSITION)
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
            else if (typeH == 3 || typeH == 15 || typeH == 31) // o-i
            {
                typeR[3] = true;      // Set inside flag (typeR 9)
                if (mtl_current == 0) // Entry from air, keep xprmat and gain unchanged
                {
                    next_current = (short)M1;                          // Next medium
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
        if (p_acc_dist_outN)
        {
            p_acc_dist_outN[ii] = (dtype)dist_refract;
            p_acc_dist_outN[ii + n_rayN] = (dtype)dist_geo;
        }
        if (p_resolved_typeN)
            p_resolved_typeN[ii] = (uint8_t)typeR;
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
                                             const arma::Mat<float> *path_dir_prev, const arma::Mat<float> *acc_dist_in,
                                             arma::Col<short> *mtl_ind_prev_outN, arma::Col<short> *mtl_ind_current_outN,
                                             arma::Col<short> *mtl_ind_buffer_outN,
                                             arma::Col<float> *gainN, arma::Mat<float> *xprmatN,
                                             arma::Mat<float> *path_dirN, arma::Mat<float> *acc_dist_outN, std::vector<uint8_t> *resolved_typeN,
                                             const arma::u32_vec *ray_indN, double eps);

template void quadriga_lib::ray_state_update(int interaction_type, double center_frequency,
                                             const arma::Mat<double> *orig, const arma::Mat<double> *dest,
                                             const arma::Mat<double> *fbsN, const arma::Mat<double> *sbsN,
                                             const arma::u32_vec *no_interact, const arma::Col<double> *fbs_angleN,
                                             const arma::Mat<double> *normal_vecN, const std::vector<uint8_t> *out_typeN,
                                             const std::unordered_map<std::string, std::vector<double>> *mtl_prop,
                                             const arma::Col<short> *mtl_ind_fbs, const arma::Col<short> *mtl_ind_sbs,
                                             const arma::Col<short> *mtl_ind_prev_in, const arma::Col<short> *mtl_ind_current_in,
                                             const arma::Col<short> *mtl_ind_buffer_in,
                                             const arma::Mat<double> *path_dir_prev, const arma::Mat<double> *acc_dist_in,
                                             arma::Col<short> *mtl_ind_prev_outN, arma::Col<short> *mtl_ind_current_outN,
                                             arma::Col<short> *mtl_ind_buffer_outN,
                                             arma::Col<double> *gainN, arma::Mat<double> *xprmatN,
                                             arma::Mat<double> *path_dirN, arma::Mat<double> *acc_dist_outN, std::vector<uint8_t> *resolved_typeN,
                                             const arma::u32_vec *ray_indN, double eps);
