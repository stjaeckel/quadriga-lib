// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <complex>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cmath>

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
        double medium_gain(double dist, double fGHz = 1.0, double abs_cos_theta = 1.0) const
        {
            std::complex<double> eta_val = eta(fGHz) * mu(fGHz);
            double er = std::real(eta_val);
            double tan_delta = std::imag(eta_val) / er;
            double cos_delta = 1.0 / std::sqrt(1.0 + tan_delta * tan_delta);
            double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
            Delta = std::sqrt(Delta) * 0.0477135 / (fGHz * std::sqrt(er));
            double loss = dist * 8.686 / Delta;
            loss += dist * alpha * std::pow(fGHz / fRef, alphaB);
            constexpr double mass_min_path = 0.0015;
            if (m > 0.0 && dist > mass_min_path)
            {
                double mass_path = dist * abs_cos_theta * abs_cos_theta;
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
        // 'this' is the medium the ray travels in (incidence side, medium 1); 'other' is the medium it enters into / reflects off
        // (medium 2). Computes the ITU-R P.2040-1 interface coefficients and returns the interface power gain.
        // For transmission/refraction (1/2/4), the returned gain and cTE/cTM include the entered medium's lumped
        // interface_gain (att + coincidence); reflection (0/3) does not. 0.5*(|cTE|^2 + |cTM|^2) == gain in all cases.
        double interact_with(const Material &other,                         // Material that the path enters into / reflects of
                             int interaction_type,                          // 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission
                             double theta,                                  // Incidence angle
                             double fGHz,                                   // Frequency
                             std::complex<double> *cTE = nullptr,           // Out: E-field coefficient, R for reflection (0/3), T for transmission/refraction (1/2/4)
                             std::complex<double> *cTM = nullptr,           // Out: M-field coefficient
                             std::complex<double> *cos_theta2 = nullptr,    // Out: Refraction cosine (type-2 direction)
                             std::complex<double> *eta1_div_eta2 = nullptr, // Out: eta1/eta2
                             double *Snell_ratio = nullptr,                 // Out: sqrt|eta1*mu1 / eta2*mu2| (Snell ratio, type-2 direction)
                             bool *total_reflection = nullptr,              // Out: Total reflection indicator
                             bool *dense2light = nullptr,                   // Out: Dense to light medium indicator
                             bool force_tir = false) const                  // Switch to force total internal reflection
        {
            const bool is_scalar = interaction_type >= 3;
            int geometry_type = interaction_type;
            if (interaction_type == 3)
                geometry_type = 0; // scalar reflection -> reflection geometry
            if (interaction_type == 4)
                geometry_type = 1; // scalar transmission -> transmission geometry

            // Incidence cosine (fbs_angleN convention); identical to |OF . N| in ray_mesh_interact
            double abs_cos_theta = std::abs(std::cos(theta + 1.570796326794897));
            abs_cos_theta = (abs_cos_theta > 1.0) ? 1.0 : abs_cos_theta;
            double sin_theta = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta);

            // Interface permittivity (resonance included) and permeability for both media
            std::complex<double> eta1 = eta(fGHz) + eta_resonance(fGHz);
            std::complex<double> eta2 = other.eta(fGHz) + other.eta_resonance(fGHz);
            std::complex<double> mu1 = mu(fGHz);
            std::complex<double> mu2 = other.mu(fGHz);

            bool d2l = std::real(eta1 * mu1) > std::real(eta2 * mu2);

            std::complex<double> eta1_d_eta2 = (eta1 * mu1) / (eta2 * mu2);
            double snell = std::sqrt(std::abs(eta1_d_eta2));
            bool tir = is_scalar ? false : (force_tir || (snell * sin_theta >= 1.0));
            std::complex<double> ct2 = std::sqrt(1.0 - eta1_d_eta2 * sin_theta * sin_theta);

            // Admittances sqrt(eps/mu)
            std::complex<double> z1 = std::sqrt(eta1 / mu1);
            std::complex<double> z2 = std::sqrt(eta2 / mu2);

            // Reflection coefficients, ITU-R P.2040-1 eq. (31)
            std::complex<double> R_eTE = tir ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
            std::complex<double> R_eTM = R_eTE;
            double reflection_gain = tir ? 1.0 : 0.0;

            if (is_scalar)
            {
                R_eTE = (z1 * abs_cos_theta - z2 * ct2) / (z1 * abs_cos_theta + z2 * ct2);
                R_eTM = R_eTE;
                reflection_gain = std::norm(R_eTE);
            }
            else if (interaction_type == 1 || (interaction_type == 0 && !tir))
            {
                R_eTE = (z1 * abs_cos_theta - z2 * ct2) / (z1 * abs_cos_theta + z2 * ct2);
                R_eTM = (z2 * abs_cos_theta - z1 * ct2) / (z2 * abs_cos_theta + z1 * ct2);
                reflection_gain = 0.5 * (std::norm(R_eTE) + std::norm(R_eTM));
            }

            // Transmission coefficients, ITU-R P.2040-1 eq. (32)
            std::complex<double> T_eTE(0.0, 0.0), T_eTM(0.0, 0.0);
            double refraction_gain = 0.0;
            if (!tir && !is_scalar && interaction_type != 0)
            {
                T_eTE = (2.0 * z1 * abs_cos_theta) / (z1 * abs_cos_theta + z2 * ct2);
                T_eTM = (2.0 * z1 * abs_cos_theta) / (z2 * abs_cos_theta + z1 * ct2);
                refraction_gain = 0.5 * (std::norm(T_eTE) + std::norm(T_eTM));
            }

            // Scalar transmission factor: redistribute reflection/transmission energy keeping the sum at 1.
            // tf is the FBS-face material's factor: the entered material ('other') on a front hit, the
            // incidence material ('this') on a back hit, selected by the sign of theta.
            if (is_scalar)
            {
                double tf_eff = (theta >= 0.0) ? other.tf(fGHz) : tf(fGHz);
                double R0 = (reflection_gain < 0.0) ? 0.0 : ((reflection_gain > 1.0) ? 1.0 : reflection_gain);
                double refl = (tf_eff >= 0.0) ? R0 * (1.0 - tf_eff) : R0 + (1.0 - R0) * (-tf_eff);
                double R_phase = std::arg(R_eTE);
                double T_phase = std::arg(1.0 + R_eTE);
                R_eTE = std::polar(std::sqrt(refl), R_phase);
                R_eTM = R_eTE;
                T_eTE = std::polar(std::sqrt(1.0 - refl), T_phase);
                T_eTM = T_eTE;
                reflection_gain = refl;
                refraction_gain = 1.0 - refl;
            }

            // EM dense->light transmission: full pass-through
            if (geometry_type == 1 && d2l && !is_scalar)
            {
                T_eTE = std::complex<double>(1.0, 0.0);
                T_eTM = std::complex<double>(1.0, 0.0);
                refraction_gain = 1.0;
                reflection_gain = 0.0;
            }

            // Select coefficient set and interface power gain
            std::complex<double> coeff_TE = (geometry_type == 0) ? R_eTE : T_eTE;
            std::complex<double> coeff_TM = (geometry_type == 0) ? R_eTM : T_eTM;
            double gain;

            if (geometry_type == 0) // reflection (types 0, 3)
                gain = reflection_gain;
            else if (interaction_type == 1) // EM transmission: energy-conserving forward beam, power = 1 - R
            {
                gain = 1.0 - reflection_gain;
                if (refraction_gain > 0.0) // fold the (1-R)/refraction_gain rescale into the coefficients
                {
                    double s = std::sqrt(gain / refraction_gain);
                    coeff_TE *= s;
                    coeff_TM *= s;
                }
            }
            else // EM refraction (2): raw Fresnel power; scalar transmission (4): refraction_gain == 1 - refl
                gain = refraction_gain;

            // Adjust transmission / refraction by interface gain (att + coincidence)
            if (geometry_type != 0)
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
            if (dense2light)
                *dense2light = d2l;

            return gain;
        }

        // Analytic thin-slab (Fabry-Perot) factor S = 1 / (1 - r_near * r_far * phi^2)
        // Returns S, or a NaN complex on re-emit: when parallel_ok is false (known wedge/edge), when the
        // round-trip amplitude rho falls below eps, or when the denominator sits near the pole. Callers
        // test the result with std::isnan(std::real(S)).
        std::complex<double> slab_airy_factor(const Material &near,    // Material on the far side of the interface being processed (r_near, slab side)
                                              const Material &far,     // Material on the far side of the opposite interface (r_far)
                                              double theta,            // Incidence angle
                                              double dist,             // One-way in-slab path d(orig, fbs)
                                              double fGHz = 1.0,       // Frequency
                                              double eps = 0.15,       // Resolve threshold
                                              bool parallel_ok = true) // Set false if near/far interface are known not-parallel
        {
            const std::complex<double> nan_c(std::nan(""), std::nan(""));
            if (!parallel_ok) // known wedge/edge -> re-emit
                return nan_c;

            const double c0 = 299792458.0;
            const double omega = 2.0 * 3.14159265358979323846 * fGHz * 1e9;

            // Slab medium (= *this). eta_if includes the resonance pole (interface / Fresnel); eta_med excludes it (medium path / phase).
            bool slab_is_air = same_as(Material());
            std::complex<double> eta_s_if = eta(fGHz) + eta_resonance(fGHz);
            std::complex<double> mu_s = mu(fGHz);
            std::complex<double> eta_s_med = eta(fGHz);

            // Incidence cosine (fbs_angleN convention)
            double abs_cos = std::abs(std::cos(theta + 1.570796326794897));
            abs_cos = (abs_cos > 1.0) ? 1.0 : abs_cos;
            double sin2 = 1.0 - abs_cos * abs_cos;

            // Fresnel (TE) amplitude reflection at slab|adjacent from the slab side, with tf folded into
            // the magnitude and the Fresnel phase preserved. Returns r and R = |r|^2.
            auto fresnel_r = [&](const Material &adj, std::complex<double> &r, double &R)
            {
                std::complex<double> eta_a_if = adj.eta(fGHz) + adj.eta_resonance(fGHz);
                std::complex<double> mu_a = adj.mu(fGHz);
                std::complex<double> z1 = std::sqrt(eta_s_if / mu_s); // slab admittance
                std::complex<double> z2 = std::sqrt(eta_a_if / mu_a); // adjacent admittance
                std::complex<double> ratio = (eta_s_if * mu_s) / (eta_a_if * mu_a);
                std::complex<double> cos_t2 = std::sqrt(1.0 - ratio * sin2);
                std::complex<double> r_te = (z1 * abs_cos - z2 * cos_t2) / (z1 * abs_cos + z2 * cos_t2);
                double R0 = std::norm(r_te);

                // tf of the face owner: the slab if solid, the adjacent solid for an air gap
                double Reff = (slab_is_air ? adj : *this).apply_tf(R0, fGHz);

                r = std::polar(std::sqrt(Reff), std::arg(r_te));
                R = Reff;
            };

            std::complex<double> r_near, r_far;
            double R_near = 0.0, R_far = 0.0;
            fresnel_r(near, r_near, R_near);
            fresnel_r(far, r_far, R_far);

            // One-way in-slab propagation phi: magnitude from the full medium_gain (dielectric + alpha +
            // mass, with the mass-law angle factor evaluated at the actual incidence cosine so that
            // dist * cos^2 = d * cos(theta) recovers the surface mass), phase from the resonance-excluded
            // permittivity only. Air slab -> lossless, unit index.
            double gL = medium_gain(dist, fGHz, abs_cos);
            double n_re = std::real(std::sqrt(eta_s_med * mu_s)); // real refractive index
            double abs_phi = std::sqrt((gL < 0.0) ? 0.0 : gL);
            double arg_phi = -(omega / c0) * n_re * dist;
            std::complex<double> phi2 = std::polar(abs_phi * abs_phi, 2.0 * arg_phi); // phi^2
            std::complex<double> denom = std::complex<double>(1.0, 0.0) - r_near * r_far * phi2;

            // Survival gate: rho^2 = R_near * R_far * medium_gain(2L)
            double g2L = medium_gain(2.0 * dist, fGHz, abs_cos);
            double rr = R_near * R_far;
            rr = (rr < 0.0) ? 0.0 : rr;
            g2L = (g2L < 0.0) ? 0.0 : g2L;
            double rho = std::sqrt(rr * g2L);

            // Survival + near-pole clamp -> re-emit
            if (rho < eps || std::abs(denom) < 1.0e-2)
                return nan_c;

            return std::complex<double>(1.0, 0.0) / denom;
        }
    };

    // Mirror reflection direction: d = u - 2*c*n, c = clamp(u.n)
    inline void qd_reflect(double Ux, double Uy, double Uz,
                           double Nx, double Ny, double Nz,
                           double &Dx, double &Dy, double &Dz)
    {
        double c = Ux * Nx + Uy * Ny + Uz * Nz;
        c = (c < -1.0) ? -1.0 : (c > 1.0 ? 1.0 : c);
        double a = 2.0 * c;
        Dx = Ux - a * Nx, Dy = Uy - a * Ny, Dz = Uz - a * Nz;
    }

    // Snell refraction direction: normalize(eta*u + (eta*cos_in - Re(cos_theta2))*n)
    inline void qd_refract(double Ux, double Uy, double Uz,
                           double Nx, double Ny, double Nz,
                           double eta, double cos_in, std::complex<double> cos_theta2,
                           double &Dx, double &Dy, double &Dz)
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

    // Per-hit in-medium charge: incidence-side leg (if ray starts inside) and transmissive offset leg
    inline double qd_hit_charges(const Material &M1, const Material &M2,
                                 bool ray_starts_inside, int geometry_type,
                                 double OF_length, double ray_offset,
                                 double fGHz, double abs_cos_theta)
    {
        double gain = 1.0;
        if (ray_starts_inside)
        {
            double thickness = (geometry_type == 0) ? OF_length + ray_offset : OF_length;
            gain *= M1.medium_gain(thickness, fGHz, abs_cos_theta);
        }
        if (geometry_type != 0)
            gain *= M2.medium_gain(ray_offset, fGHz);
        return gain;
    }
}

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# medium_gain
Linear gain of a ray traversing a homogeneous lossy medium

- Computes `g = 10^(-A/10)`, where `A` [dB] is the total attenuation accumulated over a path
  of length `dist` inside the medium. The per-meter loss combines two contributions:
  - Conductivity-based loss from the complex permittivity model of ITU-R P.2040-1: `ε_r = a·(f/fRef)^b`,
    `σ = c·(f/fRef)^d`. These give an gain distance `Δ` and a per-meter power loss `8.686 / Δ` dB/m.
  - Distance absorption of the form `α·(f/fRef)^αB` dB/m, intended to model excess loss not captured
    by `σ` (e.g. foliage, scattering media).
- The penetration-loss columns (`att`, `attB`) of `mtl_prop` are not used — they describe
  thin-slab transmission loss, not propagation through a finite-thickness medium.

## Declaration:
```
dtype quadriga_lib::medium_gain(
    const arma::Mat<dtype> &mtl_prop,
    arma::uword iM,
    dtype dist,
    dtype fGHz);
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
# ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

- Computes interaction of plane waves with planar interfaces between homogeneous isotropic media.
- Supports beam-based modeling via triangular ray tubes (`trivec`, `tridir`).
- Face side determined by vertex order; CCW winding = front, CW = back (right-hand rule);
  front-side hit with FBS≠SBS → air-to-media; back-side hit with FBS≠SBS → media-to-air;
  FBS=SBS with opposing normals → media-to-media.
- Rays with `fbs_ind = 0` (no interaction) are omitted from output, so `n_rayN ≤ n_ray`.
- Output direction encoding (spherical/Cartesian) matches input `tridir` format.
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves).
- Types 3–4 (scalar) use TE-only reflection with no total internal reflection, suitable for acoustic
  simulation with impedance-mapped material parameters (ε derived from Z).
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

## Declaration:
```
void quadriga_lib::ray_mesh_interact(
    int interaction_type,
    dtype center_frequency,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *fbs,
    const arma::Mat<dtype> *sbs,
    const arma::Mat<dtype> *mesh,
    const arma::uvec *mtl_ind,
    const std::unordered_map<std::string, td::vector<dtype>> *mtl_prop,
    const arma::u32_vec *fbs_ind,
    const arma::u32_vec *sbs_ind,
    const arma::Mat<dtype> *trivec = nullptr,
    const arma::Mat<dtype> *tridir = nullptr,
    const arma::Col<dtype> *orig_length = nullptr,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *trivecN = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    arma::Col<dtype> *orig_lengthN = nullptr,
    arma::Col<dtype> *fbs_angleN = nullptr,
    arma::Col<dtype> *thicknessN = nullptr,
    arma::Col<dtype> *edge_lengthN = nullptr,
    arma::Mat<dtype> *normal_vecN = nullptr,
    arma::s32_vec *out_typeN = nullptr);
```

## Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission
- **`center_frequency`** — Center frequency
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`fbs`**, **`sbs`** — First/second interaction points in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see [[obj_file_read]]; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`. 0 = face has no material (air). NULL → all faces treated as air.
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`. NULL → air defaults used.
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian
- **`orig_length`** *(optional)* — Accumulated path length at origin; `[n_ray]`, default 0

## Outputs:
- **`origN`** — New origins after interaction (offset 0.001 m along travel direction); `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`gainN`** — Interaction gain (linear scale, includes in-medium attenuation, excludes FSPL); averaged over TE/TM polarizations for types 0–2, TE-only for types 3–4; `[n_rayN]`
- **`xprmatN`** — For types 0–2: polarization transfer matrix, interleaved complex `[ReVV ImVV ReVH ImVH ReHV ImHV ReHH ImHH]`;
  includes interaction gain, TE/TM coefficients, incidence plane orientation, in-medium attenuation (excludes FSPL);
  `[n_rayN, 8]`. For types 3–4 (scalar): `[Re Im 0 0 0 0 0 0]` where Re+jIm is the scalar pressure coefficient including
  in-medium attenuation; `[n_rayN, 8]`.
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if inputs not provided
- **`orig_lengthN`** — Path length from `orig` to `origN`, added to input `orig_length` if given; `[n_rayN]`
- **`fbs_angleN`** — Incidence angle at FBS in rad; `[n_rayN]`
- **`thicknessN`** — Material thickness (FBS-to-SBS distance); `[n_rayN]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (∞ if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code; `[n_rayN]`<br><br>
   | Code  | Description                                         |
   | :---: | --------------------------------------------------- |
   |   1   | Single hit, outside→inside                          |
   |   2   | Single hit, inside→outside                          |
   |   3   | Single hit, inside→outside, total reflection        |
   |   4   | Media-to-media, M2 hit first                        |
   |   5   | Media-to-media, M1 hit first                        |
   |   6   | Media-to-media, M1 hit first, total reflection      |
   |   7   | Overlapping faces, outside→inside                   |
   |   8   | Overlapping faces, inside→outside                   |
   |   9   | Overlapping faces, inside→outside, total reflection |
   |  10   | Edge hit, outside→inside→outside                    |
   |  11   | Edge hit, inside→outside→inside                     |
   |  12   | Edge hit, inside→outside→inside, total reflection   |
   |  13   | Edge hit, outside→inside                            |
   |  14   | Edge hit, inside→outside                            |
   |  15   | Edge hit, inside→outside, total reflection          |

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
                                     const arma::Mat<dtype> *fbs,
                                     const arma::Mat<dtype> *sbs,
                                     const arma::Mat<dtype> *mesh,
                                     const arma::uvec *mtl_ind,
                                     const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
                                     const arma::u32_vec *fbs_ind,
                                     const arma::u32_vec *sbs_ind,
                                     const arma::Mat<dtype> *trivec,
                                     const arma::Mat<dtype> *tridir,
                                     const arma::Col<dtype> *orig_length,
                                     arma::Mat<dtype> *origN,
                                     arma::Mat<dtype> *destN,
                                     arma::Col<dtype> *gainN,
                                     arma::Mat<dtype> *xprmatN,
                                     arma::Mat<dtype> *trivecN,
                                     arma::Mat<dtype> *tridirN,
                                     arma::Col<dtype> *orig_lengthN,
                                     arma::Col<dtype> *fbs_angleN,
                                     arma::Col<dtype> *thicknessN,
                                     arma::Col<dtype> *edge_lengthN,
                                     arma::Mat<dtype> *normal_vecN,
                                     arma::s32_vec *out_typeN)
{

    // Internal documentation: out_typeN mapping logic
    //   No | θF<0 | θS<0 | dFS=0 | TotRef | iSBS=0 | NF=-NS | NF=NS | startIn | endIn | Meaning
    //   ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
    //    0 |      |      |       |        |        |        |       |         |       | Undefined
    //    1 |   no |  N/A |    no |    N/A |    yes |    N/A |   N/A |      no |   yes | Single Hit o-i
    //    2 |  yes |  N/A |    no |     no |    yes |    N/A |   N/A |     yes |    no | Single Hit i-o
    //    3 |  yes |  N/A |    no |    yes |    yes |    N/A |   N/A |     yes |    no | Single Hit i-o, TR
    //   ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
    //    4 |   no |  yes |   yes |     no |     no |    yes |    no |     yes |   yes | M2M, M2 hit first
    //    5 |  yes |   no |   yes |     no |     no |    yes |    no |     yes |   yes | M2M, M1 hit first
    //    6 |  yes |   no |   yes |    yes |     no |    yes |    no |     yes |   yes | M2M, M1 hit first, TR
    //   ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
    //    7 |   no |   no |   yes |    N/A |     no |     no |   yes |      no |   yes | Overlapping Faces, o-i
    //    8 |  yes |  yes |   yes |     no |     no |     no |   yes |     yes |    no | Overlapping Faces, i-o
    //    9 |  yes |  yes |   yes |    yes |     no |     no |   yes |     yes |    no | Overlapping Faces, i-o, TR
    //   ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
    //   10 |   no |  yes |   yes |    N/A |     no |     no |    no |      no |    no | Edge Hit, o-i-o
    //   11 |  yes |   no |   yes |     no |     no |     no |    no |     yes |   yes | Edge Hit, i-o-i
    //   12 |  yes |   no |   yes |    yes |     no |     no |    no |     yes |   yes | Edge Hit, i-o-i, TR
    //   13 |   no |   no |   yes |    N/A |     no |     no |    no |      no |   yes | Edge Hit, o-i
    //   14 |  yes |  yes |   yes |     no |     no |     no |    no |     yes |    no | Edge Hit, i-o
    //   15 |  yes |  yes |   yes |    yes |     no |     no |    no |     yes |    no | Edge Hit, i-o, TR

    // Ray offset is used to detect co-location of points, value in meters
    const double ray_offset = 0.001;

    // Check interaction_type
    if (interaction_type < 0 || interaction_type > 4)
        throw std::invalid_argument("Interaction type must be either (0) EM Reflection, (1) EM Transmission, (2) EM Refraction, (3) Scalar Reflection, (4) Scalar Transmission");

    bool is_scalar = interaction_type >= 3;

    int geometry_type = interaction_type;
    if (interaction_type == 3) // scalar reflection → reflection geometry
        geometry_type = 0;
    if (interaction_type == 4) // scalar transmission → transmission geometry
        geometry_type = 1;

    // Frequency in GHz
    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    double fGHz = (double)center_frequency * 1.0e-9;

    // Check for NULL pointers
    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (dest == nullptr)
        throw std::invalid_argument("Input 'dest' cannot be NULL.");
    if (fbs == nullptr)
        throw std::invalid_argument("Input 'fbs' cannot be NULL.");
    if (sbs == nullptr)
        throw std::invalid_argument("Input 'sbs' cannot be NULL.");
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (fbs_ind == nullptr)
        throw std::invalid_argument("Input 'fbs_ind' cannot be NULL.");
    if (sbs_ind == nullptr)
        throw std::invalid_argument("Input 'sbs_ind' cannot be NULL.");

    // Check for correct number of columns
    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    if (dest->n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");
    if (fbs->n_cols != 3)
        throw std::invalid_argument("Input 'fbs' must have 3 columns containing x,y,z coordinates.");
    if (sbs->n_cols != 3)
        throw std::invalid_argument("Input 'sbs' must have 3 columns containing x,y,z coordinates.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");

    const arma::uword n_ray = orig->n_rows;  // Number of rays
    const arma::uword n_mesh = mesh->n_rows; // Number of mesh elements

    // Check for correct number of rows
    if (dest->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");
    if (fbs->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'fbs' dont match.");
    if (sbs->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'sbs' dont match.");
    if (mtl_ind != nullptr && !mtl_ind->is_empty() && mtl_ind->n_elem != n_mesh)
        throw std::invalid_argument("Length of 'mtl_ind' must match the number of mesh faces.");
    if (fbs_ind->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'fbs_ind' does not match number of rows in 'orig'.");
    if (sbs_ind->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'sbs_ind' does not match number of rows in 'orig'.");

    // Check input data for ray tube
    int use_ray_tube = 0;
    if (trivec != nullptr && !trivec->is_empty())
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
    else if (tridir != nullptr && !tridir->is_empty())
        throw std::invalid_argument("In order to use ray tubes, both 'trivec' and 'tridir' must be given.");

    // Check for 'orig_length'
    if (orig_length != nullptr && !orig_length->is_empty() && orig_length->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'orig_length' does not match number of rows in 'orig'.");

    // Get input pointers
    const dtype *p_orig = orig->memptr();
    const dtype *p_dest = dest->memptr();
    const dtype *p_fbs = fbs->memptr();
    const dtype *p_sbs = sbs->memptr();
    const dtype *p_mesh = mesh->memptr();
    const unsigned *p_fbs_ind = fbs_ind->memptr();
    const unsigned *p_sbs_ind = sbs_ind->memptr();
    const dtype *p_trivec = (trivec == nullptr) ? nullptr : trivec->memptr();
    const dtype *p_tridir = (tridir == nullptr) ? nullptr : tridir->memptr();
    const dtype *p_orig_length = (orig_length == nullptr) ? nullptr : orig_length->memptr();

    // Resolve material columns once; air (empty) table when no material model is supplied.
    const arma::uword *p_mtl_ind = (mtl_ind == nullptr || mtl_ind->is_empty()) ? nullptr : mtl_ind->memptr();
    MaterialCols<dtype> cols = (mtl_prop != nullptr) ? MaterialCols<dtype>(*mtl_prop) : MaterialCols<dtype>();
    if (p_mtl_ind != nullptr && (arma::uword)mtl_ind->max() > cols.n_mtl)
        throw std::invalid_argument("Values in 'mtl_ind' exceed the number of materials in 'mtl_prop'.");

    // Get number of output rays and build output ray index
    // - Only consider rays that have at least one interaction with the mesh, i.e. 'fbs_ind != 0'
    unsigned n_rayN_u = 0;
    unsigned *output_ray_index = new unsigned[n_ray]; // 1-based
    for (size_t i_ray = 0; i_ray < n_ray; ++i_ray)    // Ray loop
        if (p_fbs_ind[i_ray] == 0)                    // No hit
            output_ray_index[i_ray] = 0;
        else if (p_fbs_ind[i_ray] > n_mesh) // Invalid, must be 1 ... n_mesh (1-based index)
            throw std::invalid_argument("Some values in 'fbs_ind' exceed number of mesh elements.");
        else if (p_sbs_ind[i_ray] > n_mesh)
            throw std::invalid_argument("Some values in 'sbs_ind' exceed number of mesh elements.");
        else // Store value
            output_ray_index[i_ray] = ++n_rayN_u;

    const arma::uword n_rayN = (arma::uword)n_rayN_u;

    // Allocate output memory, if needed
    if (origN != nullptr && (origN->n_rows != n_rayN || origN->n_cols != 3))
        origN->set_size(n_rayN, 3);

    if (destN != nullptr && (destN->n_rows != n_rayN || destN->n_cols != 3))
        destN->set_size(n_rayN, 3);

    if (gainN != nullptr && gainN->n_elem != n_rayN)
        gainN->set_size(n_rayN);

    if (xprmatN != nullptr && (xprmatN->n_rows != n_rayN || xprmatN->n_cols != 8))
        xprmatN->set_size(n_rayN, 8);

    if (trivecN != nullptr && use_ray_tube && (trivecN->n_rows != n_rayN || trivecN->n_cols != 9))
        trivecN->set_size(n_rayN, 9);
    else if (trivecN != nullptr && !use_ray_tube && !trivecN->is_empty())
        trivecN->reset();

    if (tridirN != nullptr && use_ray_tube == 1 && (tridirN->n_rows != n_rayN || tridirN->n_cols != 6))
        tridirN->set_size(n_rayN, 6);
    else if (tridirN != nullptr && use_ray_tube == 2 && (tridirN->n_rows != n_rayN || tridirN->n_cols != 9))
        tridirN->set_size(n_rayN, 9);
    else if (tridirN != nullptr && !use_ray_tube && !tridirN->is_empty())
        tridirN->reset();

    if (orig_lengthN != nullptr && orig_lengthN->n_elem != n_rayN)
        orig_lengthN->set_size(n_rayN);

    if (fbs_angleN != nullptr && fbs_angleN->n_elem != n_rayN)
        fbs_angleN->set_size(n_rayN);

    if (thicknessN != nullptr && thicknessN->n_elem != n_rayN)
        thicknessN->set_size(n_rayN);

    if (edge_lengthN != nullptr && edge_lengthN->n_elem != n_rayN)
        edge_lengthN->set_size(n_rayN);

    if (normal_vecN != nullptr && (normal_vecN->n_rows != n_rayN || normal_vecN->n_cols != 6))
        normal_vecN->set_size(n_rayN, 6);

    if (out_typeN != nullptr && out_typeN->n_elem != n_rayN)
        out_typeN->set_size(n_rayN);

    // Get output pointers
    dtype *p_origN = (origN == nullptr) ? nullptr : origN->memptr();
    dtype *p_destN = (destN == nullptr) ? nullptr : destN->memptr();
    dtype *p_gainN = (gainN == nullptr) ? nullptr : gainN->memptr();
    dtype *p_xprmatN = (xprmatN == nullptr) ? nullptr : xprmatN->memptr();
    dtype *p_trivecN = (trivecN == nullptr) ? nullptr : trivecN->memptr();
    dtype *p_tridirN = (tridirN == nullptr) ? nullptr : tridirN->memptr();
    dtype *p_orig_lengthN = (orig_lengthN == nullptr) ? nullptr : orig_lengthN->memptr();
    dtype *p_fbs_angleN = (fbs_angleN == nullptr) ? nullptr : fbs_angleN->memptr();
    dtype *p_thicknessN = (thicknessN == nullptr) ? nullptr : thicknessN->memptr();
    dtype *p_edge_lengthN = (edge_lengthN == nullptr) ? nullptr : edge_lengthN->memptr();
    dtype *p_normal_vecN = (normal_vecN == nullptr) ? nullptr : normal_vecN->memptr();
    int *p_out_typeN = (out_typeN == nullptr) ? nullptr : out_typeN->memptr();

    // Only calculate ray tube if it is required in the output
    if (use_ray_tube && p_trivecN == nullptr && p_tridirN == nullptr)
        use_ray_tube = 0;

#pragma omp parallel for
    for (long long i_ray = 0; i_ray < (long long)n_ray; ++i_ray) // Ray loop
    {
        if (p_fbs_ind[i_ray] == 0) // Skip non-hits
            continue;

        size_t iRx = (size_t)i_ray;               // Ray x-index
        size_t iRy = iRx + n_ray;                 // Ray y-index
        size_t iRz = iRy + n_ray;                 // Ray z-index
        size_t iFBS = (size_t)p_fbs_ind[iRx] - 1; // Mesh FBS index, 0-based
        size_t iSBS = (size_t)p_sbs_ind[iRx];     // Mesh SBS index, 1-based

        // Material indices for FBS and SBS faces (0 if no material table)
        arma::uword iMF = (p_mtl_ind == nullptr) ? 0 : (arma::uword)p_mtl_ind[iFBS];
        arma::uword iMS = (p_mtl_ind == nullptr || iSBS == 0) ? 0 : (arma::uword)p_mtl_ind[iSBS - 1];

        double Ox = (double)p_orig[iRx], Oy = (double)p_orig[iRy], Oz = (double)p_orig[iRz]; // Origin position
        double Dx = (double)p_dest[iRx], Dy = (double)p_dest[iRy], Dz = (double)p_dest[iRz]; // Destination position
        double Fx = (double)p_fbs[iRx], Fy = (double)p_fbs[iRy], Fz = (double)p_fbs[iRz];    // FBS position
        double Sx = (double)p_sbs[iRx], Sy = (double)p_sbs[iRy], Sz = (double)p_sbs[iRz];    // SBS position
        double scl = 0.0;                                                                    // Scaling factor (reused)

        // Calculate normalized vector pointing from the origin to the FBS
        double OFx = Fx - Ox, OFy = Fy - Oy, OFz = Fz - Oz;              // Vector from origin to FBS (OF)
        double OF_length = std::sqrt(OFx * OFx + OFy * OFy + OFz * OFz); // Length of vector OF
        if (OF_length < ray_offset)                                      // Origin and FBS are co-located (rare case)
            OFx = Dx - Ox, OFy = Dy - Oy, OFz = Dz - Oz,                 // Assume that Destination is the FBS
                scl = 1.0 / std::sqrt(OFx * OFx + OFy * OFy + OFz * OFz);
        else
            scl = 1.0 / OF_length;
        OFx *= scl, OFy *= scl, OFz *= scl;

        // Calculate the length of the vector from FBS to SBS
        double FSx = Sx - Fx, FSy = Sy - Fy, FSz = Sz - Fz;              // Vector pointing from FBS to SBS
        double FS_length = std::sqrt(FSx * FSx + FSy * FSy + FSz * FSz); // Length of FS

        // Surface normal vector of the FBS mesh element calculated by taking the vector cross product of two edges of the triangle
        // Note: Order of the vertices determines side (front or back) of the element
        double V1x = (double)p_mesh[iFBS],
               V1y = (double)p_mesh[iFBS + n_mesh],
               V1z = (double)p_mesh[iFBS + 2 * n_mesh];
        double E1x = (double)p_mesh[iFBS + 3 * n_mesh] - V1x,
               E1y = (double)p_mesh[iFBS + 4 * n_mesh] - V1y,
               E1z = (double)p_mesh[iFBS + 5 * n_mesh] - V1z;
        double E2x = (double)p_mesh[iFBS + 6 * n_mesh] - V1x,
               E2y = (double)p_mesh[iFBS + 7 * n_mesh] - V1y,
               E2z = (double)p_mesh[iFBS + 8 * n_mesh] - V1z;
        double Nx = E1y * E2z - E1z * E2y, Ny = E1z * E2x - E1x * E2z, Nz = E1x * E2y - E1y * E2x; // Mesh surface normal
        scl = 1.0 / std::sqrt(Nx * Nx + Ny * Ny + Nz * Nz), Nx *= scl, Ny *= scl, Nz *= scl;       // Normalize to 1

        // Calculate incidence angle between surface of the mesh element at FBS and incoming ray
        double cos_theta = OFx * Nx + OFy * Ny + OFz * Nz;                           // Angle between normal vector and incoming ray
        cos_theta = (cos_theta < -1.0) ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta); // Boundary fix
        double theta = std::acos(cos_theta) - 1.570796326794897;                     // Angle between face and incoming ray, negative values illuminate back side

        // Calculate normal vector of the SBS mesh element, if needed
        double Mx = 0.0, My = 0.0, Mz = 0.0; // SBS normal vector
        double theta_sbs = 0.0;
        if (iSBS != 0 && (FS_length < ray_offset || p_normal_vecN != nullptr))
        {
            V1x = (double)p_mesh[iSBS - 1],
            V1y = (double)p_mesh[iSBS - 1 + n_mesh],
            V1z = (double)p_mesh[iSBS - 1 + 2 * n_mesh];
            E1x = (double)p_mesh[iSBS - 1 + 3 * n_mesh] - V1x,
            E1y = (double)p_mesh[iSBS - 1 + 4 * n_mesh] - V1y,
            E1z = (double)p_mesh[iSBS - 1 + 5 * n_mesh] - V1z;
            E2x = (double)p_mesh[iSBS - 1 + 6 * n_mesh] - V1x,
            E2y = (double)p_mesh[iSBS - 1 + 7 * n_mesh] - V1y,
            E2z = (double)p_mesh[iSBS - 1 + 8 * n_mesh] - V1z;
            Mx = E1y * E2z - E1z * E2y, My = E1z * E2x - E1x * E2z, Mz = E1x * E2y - E1y * E2x;  // Mesh surface normal
            scl = 1.0 / std::sqrt(Mx * Mx + My * My + Mz * Mz), Mx *= scl, My *= scl, Mz *= scl; // Normalize to 1

            // Incidence angle at SBS
            theta_sbs = OFx * Mx + OFy * My + OFz * Mz;
            theta_sbs = (theta_sbs < -1.0) ? -1.0 : (theta_sbs > 1.0 ? 1.0 : theta_sbs);
            theta_sbs = std::acos(theta_sbs) - 1.570796326794897;
        }

        // Determine the type of the interaction
        int out_type = (theta >= 0.0) ? 1 : (theta < 0.0 ? 2 : 0); // Output type (0 = undefined, 1 = outside to inside, 2 = inside to outside)
        bool material_to_material = false;                         // Assume no material to material transition
        bool ray_starts_inside = theta < 0.0;                      // Hitting a face back side at the FBS is a certain sign
        if (FS_length < ray_offset && iSBS != 0)                   // Two colocated faces
        {
            const double lim = 1.0e-4;
            if (std::abs(Nx + Mx) < lim && std::abs(Ny + My) < lim && std::abs(Nz + Mz) < lim) // Opposing normal vectors = material to material transition
                material_to_material = true, ray_starts_inside = true,
                out_type = (theta >= 0.0) ? 4 : (theta < 0.0 ? 5 : 0);
            else if (std::abs(Nx - Mx) < lim && std::abs(Ny - My) < lim && std::abs(Nz - Mz) < lim) // Equal normal vectors = overlapping or duplicate faces
                out_type = (theta >= 0.0) ? 7 : (theta < 0.0 ? 8 : 0);
            else if (theta >= 0.0 && theta_sbs <= 0.0)
                out_type = 10; // Edge Hit, o-i-o
            else if (theta < 0.0 && theta_sbs >= 0.0)
                out_type = 11; // Edge Hit, i-o-i
            else if ((theta >= 0.0 && theta_sbs >= 0.0))
                out_type = 13; // Edge Hit, o-i
            else if ((theta < 0.0 && theta_sbs <= 0.0))
                out_type = 14; // Edge Hit, i-o
            else               // Undefined state
                out_type = 0;
        }

        // Flip normal vector in case of back side illumination
        if (theta < 0.0)
            Nx = -Nx, Ny = -Ny, Nz = -Nz,
            cos_theta = OFx * Nx + OFy * Ny + OFz * Nz,
            cos_theta = (cos_theta < -1.0) ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta);

        // Limit value to 0 ... 1 for calculating reflection and transmission coefficients
        double abs_cos_theta = std::abs(cos_theta);

        // Incidence-side (medium 1) and entered/reflected-off (medium 2) materials, default air.
        // 1-based: Material(cols, 0) -> air. M2 always carries the FBS-face material whose
        // interface_gain is the transition gain (front: iMF; back: air, or iMS for M2M).
        Material M1, M2;
        if (theta >= 0.0) // front hit: entered material = FBS face (iMF)
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
        M1.interact_with(M2, interaction_type, theta, fGHz, &cTE, &cTM, &cos_theta2, &eta1_div_eta2, &eta, &total_reflection);
        bool tir_central = total_reflection; // pre-ray-tube TIR state
        // Calculate the center path direction after medium interaction (normalized to length 1)
        double FDx = Dx - Fx, FDy = Dy - Fy, FDz = Dz - Fz;              // Vector from FBS to destination
        double FD_length = std::sqrt(FDx * FDx + FDy * FDy + FDz * FDz); // Length of path from FBS to destination

        if (geometry_type == 0) // Reflection, normalized by default
            qd_reflect(OFx, OFy, OFz, Nx, Ny, Nz, FDx, FDy, FDz);
        else if (geometry_type == 1)         // Transmission without refraction
            FDx = OFx, FDy = OFy, FDz = OFz; // New path direction = same as incoming ray, already normalized
        else                                 // Refraction
            qd_refract(OFx, OFy, OFz, Nx, Ny, Nz, eta, abs_cos_theta, cos_theta2, FDx, FDy, FDz);

        // Update origin and direction of the ray tube vertices
        double p_trivec_tmp[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double p_tridir_tmp[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double edge_length_tmp = 0.0;
        if (use_ray_tube)
        {
            // Process each vertex-ray separately
            for (int iTube = 1; iTube <= 3; ++iTube)
            {
                // Load origin and direction
                double Tx = Ox, Ty = Oy, Tz = Oz, az = 0.0, el = 0.0, Vx = 0.0, Vy = 0.0, Vz = 0.0;
                if (iTube == 1)
                {
                    Tx += (double)p_trivec[iRx], Ty += (double)p_trivec[iRy], Tz += (double)p_trivec[iRz];
                    if (use_ray_tube == 1)
                        az = (double)p_tridir[iRx], el = (double)p_tridir[iRy];
                    else
                        Vx = (double)p_tridir[iRx], Vy = (double)p_tridir[iRy], Vz = (double)p_tridir[iRz];
                }
                else if (iTube == 2)
                {
                    Tx += (double)p_trivec[iRx + 3 * n_ray], Ty += (double)p_trivec[iRx + 4 * n_ray], Tz += (double)p_trivec[iRx + 5 * n_ray];
                    if (use_ray_tube == 1)
                        az = (double)p_tridir[iRx + 2 * n_ray], el = (double)p_tridir[iRx + 3 * n_ray];
                    else
                        Vx = (double)p_tridir[iRx + 3 * n_ray], Vy = (double)p_tridir[iRx + 4 * n_ray], Vz = (double)p_tridir[iRx + 5 * n_ray];
                }
                else if (iTube == 3)
                {
                    Tx += (double)p_trivec[iRx + 6 * n_ray], Ty += (double)p_trivec[iRx + 7 * n_ray], Tz += (double)p_trivec[iRx + 8 * n_ray];
                    if (use_ray_tube == 1)
                        az = (double)p_tridir[iRx + 4 * n_ray], el = (double)p_tridir[iRx + 5 * n_ray];
                    else
                        Vx = (double)p_tridir[iRx + 6 * n_ray], Vy = (double)p_tridir[iRx + 7 * n_ray], Vz = (double)p_tridir[iRx + 8 * n_ray];
                }

                // Calculate vertex ray direction (V)
                if (use_ray_tube == 1) // Spherical input
                {
                    scl = std::cos(el);
                    Vx = std::cos(az) * scl, Vy = std::sin(az) * scl, Vz = std::sin(el);
                }
                else // Cartesian input
                {
                    double scl = Vx * Vx + Vy * Vy + Vz * Vz;
                    if (std::abs(scl - 1.0) > 2e-7)
                    {
                        scl = 1.0 / std::sqrt(scl);
                        Vx *= scl, Vy *= scl, Vz *= scl; // Normalize
                    }
                }

                // Calculate intersect point of the vertex-ray with the face
                double d = ((Fx - Tx) * Nx + (Fy - Ty) * Ny + (Fz - Tz) * Nz) / (Vx * Nx + Vy * Ny + Vz * Nz); // Distance from vert. origin to face (d)
                double Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;                                   // Intersect point with face (W)

                if (d < 0.0 || d > 1.0e5) // Vertex ray does not hit face
                    edge_length_tmp = INFINITY;

                if (geometry_type == 0) // Reflection
                {
                    if (d < 0.0 || d > 1.0e5) // Ray does not hit face - use orthogonal projection on ray
                    {
                        d = ((Fx - Tx) * Vx + (Fy - Ty) * Vy + (Fz - Tz) * Vz) / (Vx * Vx + Vy * Vy + Vz * Vz);
                        Tx = Tx + Vx * d - Fx, Ty = Ty + Vy * d - Fy, Tz = Tz + Vz * d - Fz; // Scaled vertex - updates T
                        double a = 2.0 * (Tx * Nx + Ty * Ny + Tz * Nz);                      // Reflection of T on face
                        Tx -= a * Nx + ray_offset * FDx;
                        Ty -= a * Ny + ray_offset * FDy;
                        Tz -= a * Nz + ray_offset * FDz;
                    }
                    else // Use intersect point W as new vertex origin
                    {
                        Tx = Wx - Fx - ray_offset * FDx;
                        Ty = Wy - Fy - ray_offset * FDy;
                        Tz = Wz - Fz - ray_offset * FDz;
                    }

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
                    if (d < 0.0 || d > 1.0e5) // Ray does not hit face - use orthogonal projection on vertex ray
                    {
                        d = ((Fx - Tx) * Vx + (Fy - Ty) * Vy + (Fz - Tz) * Vz) / (Vx * Vx + Vy * Vy + Vz * Vz);
                        Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;
                    }

                    // Update ray tube coordinates
                    Tx = Wx - Fx - ray_offset * FDx;
                    Ty = Wy - Fy - ray_offset * FDy;
                    Tz = Wz - Fz - ray_offset * FDz;

                    // Vertex ray directions remains the same for Transmission
                    if (geometry_type == 2) // Refraction
                    {
                        double cos_thetaV = std::abs(Vx * Nx + Vy * Ny + Vz * Nz);       // Cosine of incidence angle
                        cos_thetaV = (cos_thetaV > 1.0) ? 1.0 : cos_thetaV;              // Clamp: avoids sqrt of a negative below
                        double sin_thetaV = std::sqrt(1.0 - cos_thetaV * cos_thetaV);    // Sine of incidence angle
                        total_reflection = total_reflection | (eta * sin_thetaV >= 1.0); // Check total reflection condition

                        // Refraction into medium
                        std::complex<double> cos_theta2V = std::sqrt(1.0 - eta1_div_eta2 * sin_thetaV * sin_thetaV);
                        qd_refract(Vx, Vy, Vz, Nx, Ny, Nz, eta, cos_thetaV, cos_theta2V, Vx, Vy, Vz);
                        if (use_ray_tube == 1)
                        {
                            Vz = (Vz < -1.0) ? -1.0 : (Vz > 1.0 ? 1.0 : Vz); // Boundary fix
                            az = std::atan2(Vy, Vx), el = std::asin(Vz);     // Angles
                        }
                    }
                }

                // Write new vertex ray origin and direction - convert back to dtype
                if (iTube == 1)
                {
                    p_trivec_tmp[0] = Tx, p_trivec_tmp[1] = Ty, p_trivec_tmp[2] = Tz;
                    if (use_ray_tube == 1)
                        p_tridir_tmp[0] = az, p_tridir_tmp[1] = el;
                    else
                        p_tridir_tmp[0] = Vx, p_tridir_tmp[1] = Vy, p_tridir_tmp[2] = Vz;
                }
                else if (iTube == 2)
                {
                    p_trivec_tmp[3] = Tx, p_trivec_tmp[4] = Ty, p_trivec_tmp[5] = Tz;
                    if (use_ray_tube == 1)
                        p_tridir_tmp[2] = az, p_tridir_tmp[3] = el;
                    else
                        p_tridir_tmp[3] = Vx, p_tridir_tmp[4] = Vy, p_tridir_tmp[5] = Vz;
                }
                else if (iTube == 3)
                {
                    p_trivec_tmp[6] = Tx, p_trivec_tmp[7] = Ty, p_trivec_tmp[8] = Tz;
                    if (use_ray_tube == 1)
                        p_tridir_tmp[4] = az, p_tridir_tmp[5] = el;
                    else
                        p_tridir_tmp[6] = Vx, p_tridir_tmp[7] = Vy, p_tridir_tmp[8] = Vz;
                }
            }

            // Calculate the maximum edge length
            if (p_edge_lengthN != nullptr)
            {
                double Ex = p_trivec_tmp[3] - p_trivec_tmp[0], Ey = p_trivec_tmp[4] - p_trivec_tmp[1], Ez = p_trivec_tmp[5] - p_trivec_tmp[2];
                scl = Ex * Ex + Ey * Ey + Ez * Ez;
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

        // Determine the in-medium gain
        double gain = qd_hit_charges(M1, M2, ray_starts_inside, geometry_type,
                                     OF_length, ray_offset, fGHz, abs_cos_theta);

        // Re-evaluate coefficients if the ray tube introduced TIR (type-2 tube vertices)
        if (total_reflection != tir_central)
            M1.interact_with(M2, interaction_type, theta, fGHz, &cTE, &cTM,
                             nullptr, nullptr, nullptr, nullptr, nullptr, true);

        // Read the output ray index
        size_t i_rayN = output_ray_index[iRx] - 1; // Output ray index, 0-based
        if (i_rayN >= n_rayN)                      // Just to be sure to avoid any segfaults
            throw std::invalid_argument("Something went wrong. This should never be reached!");

        // Write origN, add a small offset to prevent it from getting stuck inside the mesh element
        if (p_origN != nullptr)
        {
            p_origN[i_rayN] = dtype(Fx + ray_offset * FDx);
            p_origN[i_rayN + n_rayN] = dtype(Fy + ray_offset * FDy);
            p_origN[i_rayN + 2 * n_rayN] = dtype(Fz + ray_offset * FDz);
        }

        // Write destN
        if (p_destN != nullptr)
        {
            // Make sure the new destination is beyond the new start point
            FD_length = (FD_length <= ray_offset) ? 2.0 * ray_offset : FD_length;
            p_destN[i_rayN] = dtype(Fx + FD_length * FDx);
            p_destN[i_rayN + n_rayN] = dtype(Fy + FD_length * FDy);
            p_destN[i_rayN + 2 * n_rayN] = dtype(Fz + FD_length * FDz);
        }

        if (p_xprmatN || p_gainN)
        {
            double amplitude = std::sqrt(gain);
            double xprmat[8], pgain;
            qd_polbasis(OFx, OFy, OFz, FDx, FDy, FDz, Nx, Ny, Nz, amplitude, cTE, cTM, is_scalar, xprmat, pgain);

            if (p_xprmatN)
                for (int k = 0; k < 8; ++k)
                    p_xprmatN[i_rayN + (size_t)k * n_rayN] = (dtype)xprmat[k];
            if (p_gainN)
                p_gainN[i_rayN] = (dtype)pgain;
        }

        // Write trivecN
        if (use_ray_tube && p_trivecN != nullptr)
        {
            p_trivecN[i_rayN] = (dtype)p_trivec_tmp[0];
            p_trivecN[i_rayN + n_rayN] = (dtype)p_trivec_tmp[1];
            p_trivecN[i_rayN + 2 * n_rayN] = (dtype)p_trivec_tmp[2];
            p_trivecN[i_rayN + 3 * n_rayN] = (dtype)p_trivec_tmp[3];
            p_trivecN[i_rayN + 4 * n_rayN] = (dtype)p_trivec_tmp[4];
            p_trivecN[i_rayN + 5 * n_rayN] = (dtype)p_trivec_tmp[5];
            p_trivecN[i_rayN + 6 * n_rayN] = (dtype)p_trivec_tmp[6];
            p_trivecN[i_rayN + 7 * n_rayN] = (dtype)p_trivec_tmp[7];
            p_trivecN[i_rayN + 8 * n_rayN] = (dtype)p_trivec_tmp[8];
        }

        // Write tridirN
        if (use_ray_tube == 1 && p_tridirN != nullptr)
        {
            p_tridirN[i_rayN] = (dtype)p_tridir_tmp[0];
            p_tridirN[i_rayN + n_rayN] = (dtype)p_tridir_tmp[1];
            p_tridirN[i_rayN + 2 * n_rayN] = (dtype)p_tridir_tmp[2];
            p_tridirN[i_rayN + 3 * n_rayN] = (dtype)p_tridir_tmp[3];
            p_tridirN[i_rayN + 4 * n_rayN] = (dtype)p_tridir_tmp[4];
            p_tridirN[i_rayN + 5 * n_rayN] = (dtype)p_tridir_tmp[5];
        }
        else if (use_ray_tube == 2 && p_tridirN != nullptr)
        {
            p_tridirN[i_rayN] = (dtype)p_tridir_tmp[0];
            p_tridirN[i_rayN + n_rayN] = (dtype)p_tridir_tmp[1];
            p_tridirN[i_rayN + 2 * n_rayN] = (dtype)p_tridir_tmp[2];
            p_tridirN[i_rayN + 3 * n_rayN] = (dtype)p_tridir_tmp[3];
            p_tridirN[i_rayN + 4 * n_rayN] = (dtype)p_tridir_tmp[4];
            p_tridirN[i_rayN + 5 * n_rayN] = (dtype)p_tridir_tmp[5];
            p_tridirN[i_rayN + 6 * n_rayN] = (dtype)p_tridir_tmp[6];
            p_tridirN[i_rayN + 7 * n_rayN] = (dtype)p_tridir_tmp[7];
            p_tridirN[i_rayN + 8 * n_rayN] = (dtype)p_tridir_tmp[8];
        }

        // Write orig_lengthN
        if (p_orig_lengthN != nullptr)
            p_orig_lengthN[i_rayN] = (p_orig_length == nullptr) ? dtype(OF_length + ray_offset)
                                                                : dtype(p_orig_length[iRx] + OF_length + ray_offset);

        // Write fbs_angleN
        if (p_fbs_angleN != nullptr)
            p_fbs_angleN[i_rayN] = (dtype)theta;

        // Write thicknessN
        if (p_thicknessN != nullptr)
            p_thicknessN[i_rayN] = (dtype)FS_length;

        // Write edge_lengthN
        if (p_edge_lengthN != nullptr)
            p_edge_lengthN[i_rayN] = (dtype)edge_length_tmp;

        // Write normal_vecN
        if (p_normal_vecN != nullptr)
        {
            // FBS normal vector
            p_normal_vecN[i_rayN] = (dtype)Nx;
            p_normal_vecN[i_rayN + n_rayN] = (dtype)Ny;
            p_normal_vecN[i_rayN + 2 * n_rayN] = (dtype)Nz;
            p_normal_vecN[i_rayN + 3 * n_rayN] = (dtype)Mx;
            p_normal_vecN[i_rayN + 4 * n_rayN] = (dtype)My;
            p_normal_vecN[i_rayN + 5 * n_rayN] = (dtype)Mz;
        }

        // Write out_typeN
        if (p_out_typeN != nullptr)
            p_out_typeN[i_rayN] = (out_type != 0 && geometry_type == 2 && total_reflection) ? out_type + 1 : out_type;
    }

    // Delete ray index
    delete[] output_ray_index;
}

template void quadriga_lib::ray_mesh_interact(int interaction_type, float center_frequency,
                                              const arma::Mat<float> *orig, const arma::Mat<float> *dest, const arma::Mat<float> *fbs, const arma::Mat<float> *sbs,
                                              const arma::Mat<float> *mesh, const arma::uvec *mtl_ind,
                                              const std::unordered_map<std::string, std::vector<float>> *mtl_prop,
                                              const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
                                              const arma::Mat<float> *trivec, const arma::Mat<float> *tridir, const arma::Col<float> *orig_length,
                                              arma::Mat<float> *origN, arma::Mat<float> *destN, arma::Col<float> *gainN, arma::Mat<float> *xprmatN,
                                              arma::Mat<float> *trivecN, arma::Mat<float> *tridirN, arma::Col<float> *orig_lengthN,
                                              arma::Col<float> *fbs_angleN, arma::Col<float> *thicknessN, arma::Col<float> *edge_lengthN,
                                              arma::Mat<float> *normal_vecN, arma::s32_vec *out_typeN);

template void quadriga_lib::ray_mesh_interact(int interaction_type, double center_frequency,
                                              const arma::Mat<double> *orig, const arma::Mat<double> *dest, const arma::Mat<double> *fbs, const arma::Mat<double> *sbs,
                                              const arma::Mat<double> *mesh, const arma::uvec *mtl_ind,
                                              const std::unordered_map<std::string, std::vector<double>> *mtl_prop,
                                              const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
                                              const arma::Mat<double> *trivec, const arma::Mat<double> *tridir, const arma::Col<double> *orig_length,
                                              arma::Mat<double> *origN, arma::Mat<double> *destN, arma::Col<double> *gainN, arma::Mat<double> *xprmatN,
                                              arma::Mat<double> *trivecN, arma::Mat<double> *tridirN, arma::Col<double> *orig_lengthN,
                                              arma::Col<double> *fbs_angleN, arma::Col<double> *thicknessN, arma::Col<double> *edge_lengthN,
                                              arma::Mat<double> *normal_vecN, arma::s32_vec *out_typeN);

/*!MD
# ray_state_update
Batched inside/outside ray-state machine with analytic thin-slab (Fabry-Perot) resolution

- Corrects the per-interaction `gainN` / `xprmatN` produced by [[ray_mesh_interact]] using a tracked
  per-ray medium state, and carries that state forward. Three signed-`short` words per ray hold the
  current medium, the previous medium, and a one-slot next-transition buffer (bit-masked: `mat = w &
  0x7FFF`, `flag = w & 0x8000`).
- Ports the inside/outside state machine formerly embedded in [[calc_diffraction_gain]] and overlays
  a closed-form thin-slab factor `S` (the Airy sum) so a single coefficient captures the full
  internal multiple-reflection series of a parallel slab thin enough to matter, instead of relying on
  the tracer to follow every internal bounce.
- Called twice per interaction by the ray tracer: once for the reflection pass (`interaction_type` 0
  or 3) and once for the transmission/refraction pass (`interaction_type` 1, 2 or 4). With `S`
  suppressed (the survival gate re-emits) the transmission/refraction path reproduces
  [[calc_diffraction_gain]] bit-for-bit.

## Declaration:
```
void quadriga_lib::ray_state_update(
    int interaction_type,
    dtype center_frequency,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *fbs,
    const arma::Mat<dtype> *sbs,
    const arma::u32_vec *no_interact,
    const arma::Col<dtype> *fbs_angleN,
    const arma::s32_vec *out_typeN,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::Col<short> *mtl_ind_fbs,
    const arma::Col<short> *mtl_ind_sbs,
    const arma::Col<short> *mtl_ind_prev_in = nullptr,
    const arma::Col<short> *mtl_ind_current_in = nullptr,
    const arma::Col<short> *mtl_ind_buffer_in = nullptr,
    const arma::Mat<dtype> *normal_vecN = nullptr,
    const arma::Mat<dtype> *path_dir_prev = nullptr,
    arma::Col<short> *mtl_ind_prev_outN = nullptr,
    arma::Col<short> *mtl_ind_current_outN = nullptr,
    arma::Col<short> *mtl_ind_buffer_outN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,
    const arma::u32_vec *ray_indN = nullptr,
    const arma::Mat<dtype> *orig_correct = nullptr,
    double eps = 0.15);
```

## Inputs:
- **`interaction_type`** — 0 EM reflection, 1 EM transmission, 2 EM refraction, 3 scalar reflection, 4 scalar transmission
- **`center_frequency`** — Center frequency in [Hz]
- **`orig`**, **`dest`**, **`fbs`**, **`sbs`** — Ray origin, destination, first and second interaction points in GCS, full ray set; `[n_ray, 3]`, read at `g = ray_ind[i]`
- **`no_interact`** — Mesh-hit count per ray, full ray set; `[n_ray]`, read at `g`
- **`fbs_angleN`** — Incidence angle at FBS (ITU convention), compact set; `[n_rayN]`
- **`out_typeN`** — Interaction type code from [[ray_mesh_interact]], compact set; `[n_rayN]`
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]])
- **`mtl_ind_fbs`**, **`mtl_ind_sbs`** — Material indices M1 / M2 of the FBS / SBS faces, compact set; `[n_rayN]` (0 = air)
- **`mtl_ind_prev_in`**, **`mtl_ind_current_in`**, **`mtl_ind_buffer_in`** — Old state words,
  full ray set; `[n_ray]`, read at `g`, never written. NULL reads as state `0` (outside, no flags).
- **`normal_vecN`** — FBS and SBS normals `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`, compact set; `[n_rayN, 6]`. NULL disables the parallelism (wedge) test
- **`eps`** — Resolve threshold for the thin-slab (Fabry-Pérot) factor `S`. A slab is solved analytically only
  when its round-trip amplitude `rho = sqrt(R_near · R_far · medium_gain(slab, 2L))` reaches `eps`; below it,
  the bounce is re-emitted for the tracer to follow. Range `[0, 1]`. `eps = 0` always resolves — required
  for a forward/transmission-only run, where no reflection pass exists to carry the internal bounces. Raise it to hand more
  weak cavities back to the tracer (`eps ≈ drop_threshold^(1/N_max)`, ~0.1–0.25); `eps >= 1` disables `S`.

## Outputs:
- **`mtl_ind_prev_out`**, **`mtl_ind_current_out`**, **`mtl_ind_buffer_out`** — New state words,
  compact set; `[n_rayN]`, written at `i`. NULL skips the write. Passing all six state args NULL disables tracking —
  each interaction is corrected on its own (entry loss, TR kill, single-hit air-gap `S`); cross-interaction slab `S` and
  reflection-bounce `S` need the tracked medium.
- **`gainN`** *(in/out)* — Per-interaction gain, patched in place; `[n_rayN]`
- **`xprmatN`** *(in/out)* — Polarization transfer matrix, columns VV, HV, VH, HH (re, im per entry), patched in place; `[n_rayN, 8]`
- **`ray_ind`** — Compact-to-full ray index map; `[n_rayN]` -> `[n_ray]`; NULL = identity (`n_ray == n_rayN`)

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
                                    const arma::Mat<dtype> *fbs,
                                    const arma::Mat<dtype> *sbs,
                                    const arma::u32_vec *no_interact,
                                    const arma::Col<dtype> *fbs_angleN,
                                    const arma::s32_vec *out_typeN,
                                    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
                                    const arma::Col<short> *mtl_ind_fbs,
                                    const arma::Col<short> *mtl_ind_sbs,
                                    const arma::Col<short> *mtl_ind_prev_in,
                                    const arma::Col<short> *mtl_ind_current_in,
                                    const arma::Col<short> *mtl_ind_buffer_in,
                                    const arma::Mat<dtype> *normal_vecN,
                                    arma::Col<short> *mtl_ind_prev_out,
                                    arma::Col<short> *mtl_ind_current_out,
                                    arma::Col<short> *mtl_ind_buffer_out,
                                    arma::Col<dtype> *gainN,
                                    arma::Mat<dtype> *xprmatN,
                                    arma::u32_vec *ray_ind,
                                    double eps)
{
    // Ray offset is used to detect co-location of points, value in meters
    const double ray_offset = 0.001;

    if (interaction_type < 0 || interaction_type > 4)
        throw std::invalid_argument("Interaction type must be either (0) EM Reflection, (1) EM Transmission, (2) EM Refraction, (3) Scalar Reflection, (4) Scalar Transmission");

    if (!std::isfinite((double)center_frequency) || center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");

    if (!std::isfinite(eps) || eps < 0.0)
        throw std::invalid_argument("Input 'eps' must be finite and >= 0.");

    if (orig == nullptr || dest == nullptr || fbs == nullptr || sbs == nullptr)
        throw std::invalid_argument("Inputs 'orig', 'dest', 'fbs' and 'sbs' cannot be NULL.");
    if (out_typeN == nullptr)
        throw std::invalid_argument("Input 'out_typeN' cannot be NULL.");

    // Resolved material columns for the Material(cols, idx)
    MaterialCols<dtype> cols = (mtl_prop != nullptr) ? MaterialCols<dtype>(*mtl_prop) : MaterialCols<dtype>();

    // Validate the material map
    if (mtl_ind_fbs != nullptr && mtl_ind_fbs->n_elem != 0 && arma::uword(mtl_ind_fbs->max() & (short)0x7FFF) > cols.n_mtl)
        throw std::invalid_argument("Values in 'mtl_ind_fbs' exceed the number of materials in 'mtl_prop'.");
    if (mtl_ind_sbs != nullptr && mtl_ind_sbs->n_elem != 0 && arma::uword(mtl_ind_sbs->max() & (short)0x7FFF) > cols.n_mtl)
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

    const bool is_scalar = interaction_type >= 3;
    const bool refl_pass = (interaction_type == 0 || interaction_type == 3); // geometry 0

    const arma::uword n_rayN = out_typeN->n_elem;
    const arma::uword n_ray = orig->n_rows;

    if (orig->n_cols != 3 || dest->n_cols != 3 || fbs->n_cols != 3 || sbs->n_cols != 3)
        throw std::invalid_argument("Inputs 'orig', 'dest', 'fbs' and 'sbs' must have 3 columns.");
    if (dest->n_rows != n_ray || fbs->n_rows != n_ray || sbs->n_rows != n_ray)
        throw std::invalid_argument("Inputs 'orig', 'dest', 'fbs' and 'sbs' must have the same number of rows.");
    if (no_interact != nullptr && no_interact->n_elem != n_ray)
        throw std::invalid_argument("Input 'no_interact' must match the number of rays in 'orig'.");
    if (mtl_ind_prev_in != nullptr && mtl_ind_prev_in->n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_prev_in' must match the number of rays in 'orig'.");
    if (mtl_ind_current_in != nullptr && mtl_ind_current_in->n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_current_in' must match the number of rays in 'orig'.");
    if (mtl_ind_buffer_in != nullptr && mtl_ind_buffer_in->n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_buffer_in' must match the number of rays in 'orig'.");
    if (fbs_angleN != nullptr && fbs_angleN->n_elem != n_rayN)
        throw std::invalid_argument("Input 'fbs_angleN' must match the length of 'out_typeN'.");
    if (mtl_ind_fbs != nullptr && mtl_ind_fbs->n_elem != n_rayN)
        throw std::invalid_argument("Input 'mtl_ind_fbs' must match the length of 'out_typeN'.");
    if (mtl_ind_sbs != nullptr && mtl_ind_sbs->n_elem != n_rayN)
        throw std::invalid_argument("Input 'mtl_ind_sbs' must match the length of 'out_typeN'.");
    if (normal_vecN != nullptr && (normal_vecN->n_rows != n_rayN || normal_vecN->n_cols != 6))
        throw std::invalid_argument("Input 'normal_vecN' must have size [n_rayN, 6].");
    if (gainN != nullptr && gainN->n_elem != n_rayN)
        throw std::invalid_argument("In-out 'gainN' must match the length of 'out_typeN'.");
    if (xprmatN != nullptr && (xprmatN->n_rows != n_rayN || xprmatN->n_cols != 8))
        throw std::invalid_argument("In-out 'xprmatN' must have size [n_rayN, 8].");
    if (ray_ind != nullptr && ray_ind->n_elem != n_rayN)
        throw std::invalid_argument("Input 'ray_ind' must match the length of 'out_typeN'.");
    if (ray_ind != nullptr && ray_ind->n_elem != 0 && (arma::uword)ray_ind->max() >= n_ray)
        throw std::invalid_argument("Values in 'ray_ind' exceed the number of rays in 'orig'.");
    if (ray_ind == nullptr && n_ray != n_rayN)
        throw std::invalid_argument("Without 'ray_ind', the full and compact sets must have the same size.");

    // Allocate / size the output state arrays (compact set)
    if (mtl_ind_prev_out != nullptr && mtl_ind_prev_out->n_elem != n_rayN)
        mtl_ind_prev_out->set_size(n_rayN);
    if (mtl_ind_current_out != nullptr && mtl_ind_current_out->n_elem != n_rayN)
        mtl_ind_current_out->set_size(n_rayN);
    if (mtl_ind_buffer_out != nullptr && mtl_ind_buffer_out->n_elem != n_rayN)
        mtl_ind_buffer_out->set_size(n_rayN);

    // Input / output pointers
    const unsigned *p_no_interact = (no_interact == nullptr) ? nullptr : no_interact->memptr();
    const int *p_out_typeN = out_typeN->memptr();
    const dtype *p_fbs_angleN = (fbs_angleN == nullptr) ? nullptr : fbs_angleN->memptr();
    const dtype *p_normal_vecN = (normal_vecN == nullptr) ? nullptr : normal_vecN->memptr();
    const short *p_M1 = (mtl_ind_fbs == nullptr) ? nullptr : mtl_ind_fbs->memptr();
    const short *p_M2 = (mtl_ind_sbs == nullptr) ? nullptr : mtl_ind_sbs->memptr();
    const short *p_prev_in = (mtl_ind_prev_in == nullptr) ? nullptr : mtl_ind_prev_in->memptr();
    const short *p_cur_in = (mtl_ind_current_in == nullptr) ? nullptr : mtl_ind_current_in->memptr();
    const short *p_buf_in = (mtl_ind_buffer_in == nullptr) ? nullptr : mtl_ind_buffer_in->memptr();
    short *p_prev_out = (mtl_ind_prev_out == nullptr) ? nullptr : mtl_ind_prev_out->memptr();
    short *p_cur_out = (mtl_ind_current_out == nullptr) ? nullptr : mtl_ind_current_out->memptr();
    short *p_buf_out = (mtl_ind_buffer_out == nullptr) ? nullptr : mtl_ind_buffer_out->memptr();
    dtype *p_gainN = (gainN == nullptr) ? nullptr : gainN->memptr();
    dtype *p_xprmatN = (xprmatN == nullptr) ? nullptr : xprmatN->memptr();
    const unsigned *p_ray_ind = (ray_ind == nullptr) ? nullptr : ray_ind->memptr();

#pragma omp parallel for
    for (long long i_rayN = 0; i_rayN < (long long)n_rayN; ++i_rayN) // Interaction loop (compact set)
    {
        size_t ii = (size_t)i_rayN;
        size_t i_ray = (p_ray_ind == nullptr) ? ii : (size_t)p_ray_ind[ii]; // Full-set index

        // Old state at g (full set). Defaults to copy-through into the compact outputs at i.
        short s_prev = (p_prev_in == nullptr) ? (short)0 : p_prev_in[i_ray];
        short s_cur = (p_cur_in == nullptr) ? (short)0 : p_cur_in[i_ray];
        short s_buf = (p_buf_in == nullptr) ? (short)0 : p_buf_in[i_ray];
        int cur = s_cur & 0x7FFF;
        bool resolved = (s_cur & 0x8000) != 0;
        int buf = s_buf & 0x7FFF;
        int prev_mat = s_prev & 0x7FFF;
        bool prev_nonpar = (s_prev & 0x8000) != 0;
        short out_prev = s_prev, out_cur = s_cur, out_buf = s_buf;

        // Compact-set reads at i
        unsigned nH = (p_no_interact == nullptr) ? 1u : p_no_interact[i_ray];
        int typeH = p_out_typeN[ii];
        int M1 = (p_M1 == nullptr) ? 0 : (int)(p_M1[ii] & (short)0x7FFF);
        int M2 = (p_M2 == nullptr) ? 0 : (int)(p_M2[ii] & (short)0x7FFF);
        double theta = (p_fbs_angleN == nullptr) ? 0.0 : (double)p_fbs_angleN[ii];
        double fGHz = (double)center_frequency * 1e-9;

        // Euclidean distance between two full-set geometry rows at g
        auto distance = [&](const arma::Mat<dtype> *A, const arma::Mat<dtype> *B) -> double
        { return (double)qd_calc_length(A->at(i_ray, 0), A->at(i_ray, 1), A->at(i_ray, 2), B->at(i_ray, 0), B->at(i_ray, 1), B->at(i_ray, 2)); };

        // Wedge test: true when FBS and SBS faces sit at a real angle. No-op (false) when normals are absent or the two faces
        // are a single point. Run only at o-i entries that capture both faces (nH >= 2 types 1/7/13).
        auto fbs_sbs_not_parallel = [&]() -> bool
        {
            if (p_normal_vecN == nullptr)
                return false;
            if (!(distance(fbs, sbs) > 1.0e-6))
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

        // Build a Material for a 1-based index (0 = air). Used by SAME, TRN and the slab-factor calls.
        auto MAT = [&](int m) -> Material
        { return Material(cols, (arma::uword)(m < 1 ? 0 : m)); };

        // Same-medium test. Index 0 is the "outside / no medium" sentinel, not a material, so it
        // only matches itself; two real materials match on identical properties.
        auto SAME = [&](int a, int b) -> bool
        { return a == b || (a > 0 && b > 0 && MAT(a).same_as(MAT(b))); };

        // Medium gain shorthand: 1-based index m (0 = air -> gain 1)
        auto MED = [&](int m, double dist) -> double
        { return Material(cols, arma::uword(m < 1 ? 0 : m)).medium_gain(dist, fGHz); };

        // Transmission gain shorthand
        auto TRN = [&](int a, int b) -> double
        { return MAT(a).interact_with(MAT(b), (is_scalar ? 4 : 1), theta, fGHz); };

        // Gain / xprmat patch operations
        auto rsu_scale = [&](double cr, double ci)
        {
            if (p_xprmatN != nullptr)
                for (int k = 0; k < 4; ++k)
                {
                    size_t re_i = ii + size_t(2 * k) * n_rayN;
                    size_t im_i = ii + size_t(2 * k + 1) * n_rayN;
                    double re = (double)p_xprmatN[re_i];
                    double im = (double)p_xprmatN[im_i];
                    p_xprmatN[re_i] = (dtype)(re * cr - im * ci);
                    p_xprmatN[im_i] = (dtype)(re * ci + im * cr);
                }
            if (p_gainN != nullptr)
                p_gainN[ii] = dtype(double(p_gainN[ii]) * (cr * cr + ci * ci));
        };

        auto rsu_replace = [&](double g)
        {
            if (p_xprmatN != nullptr)
            {
                for (int c = 0; c < 8; ++c)
                    p_xprmatN[ii + (size_t)c * n_rayN] = (dtype)0.0;
                double a = std::sqrt((g < 0.0) ? 0.0 : g);
                p_xprmatN[ii] = (dtype)a; // VV_re
                if (!is_scalar)
                    p_xprmatN[ii + (size_t)6 * n_rayN] = (dtype)a; // HH_re
            }
            if (p_gainN != nullptr)
                p_gainN[ii] = (dtype)g;
        };

        auto rsu_kill = [&]()
        {
            if (p_xprmatN != nullptr)
                for (int c = 0; c < 8; ++c)
                    p_xprmatN[ii + (size_t)c * n_rayN] = (dtype)0.0;
            if (p_gainN != nullptr)
                p_gainN[ii] = (dtype)0.0;
        };

        if (refl_pass) // Reflection pass, interaction_type {0, 3}
        {
            if (resolved) // Front reflection already summed inside S
                rsu_kill();
            else if (cur == 0) // Entry / order-0 front reflection: bare Fresnel r12 (naturally |R| = 1 under TIR). IG, state copy-through.
            {
            }
            else // Internal / back reflection of a resolvable parallel slab
            {
                // Processed interface: cur|air at an i-o face, cur|iM at an i-i face (types 4/5)
                int nearM = (typeH == 5) ? M2 : ((typeH == 4) ? M1 : 0);
                double dist = distance(orig, fbs);
                std::complex<double> S = MAT(cur).slab_airy_factor(MAT(nearM), MAT(prev_mat), theta, dist, fGHz, eps, !prev_nonpar);
                if (!std::isnan(std::real(S)))
                {
                    rsu_scale(std::real(S), std::imag(S));           // IG * S
                    out_cur = (short)((cur & 0x7FFF) | (int)0x8000); // set resolved flag
                }
                // else: ordinary reflection / re-emit: IG, copy-through
            }
        }
        else // Transmission / refraction pass, interaction_type {1, 2, 4}
        {
            if (typeH == 3 || typeH == 6 || typeH == 9 || typeH == 12 || typeH == 15) // TR forward-kill: TR out-codes occur only for interaction_type == 2 and win over the resolved flag
                rsu_kill();
            else if (resolved) // Resolved-ray out-coupling: iM = the next medium for an i-i
            {
                // Resolved rows charge the in-medium loss of their INCOMING segment (the unresolved
                // entry / M2M rows charge forward, the resolving reflection charges nothing), so every
                // segment of the resolved return path is charged exactly once.
                int iM = (typeH == 5) ? M2 : M1;
                if (typeH == 2 || typeH == 8 || typeH == 14) // i-o: out-coupling t21, up-trip loss
                {
                    rsu_scale(std::sqrt(MED(cur, distance(orig, fbs))), 0.0);
                    out_cur = (short)0; // current_out <- 0, clear resolved flag
                }
                else if (typeH == 4 || typeH == 5) // i-i: stay resolved, advance medium, incoming-segment loss
                {
                    rsu_scale(std::sqrt(MED(cur, distance(orig, fbs))), 0.0);
                    out_cur = (short)((iM & 0x7FFF) | (int)0x8000); // keep resolved flag
                    out_prev = (short)cur;                          // prev_out <- old cur
                }
                // else: o-i / edges: transparent pass-through, IG, state copy-through
            }
            else if (nH == 0) // Not processed: the caller applies any whole-segment in-medium loss. Copy-through.
            {
            }
            else if ((nH == 1 && typeH == 1) || (nH == 2 && typeH == 7) || (nH == 2 && typeH == 13)) // o-i family, entry / overlapping-entry
            {
                if (cur == 0) // enter
                {
                    double dist = distance(fbs, dest);
                    dist = (dist > ray_offset) ? dist - ray_offset : 0.0; // clamped
                    rsu_scale(std::sqrt(MED(M1, dist)), 0.0);             // IG * MED
                    out_cur = (short)M1;
                    bool nonpar = (nH >= 2) && fbs_sbs_not_parallel();
                    out_prev = (short)(nonpar ? (int)0x8000 : 0); // prev <- 0, +flag
                }
                else // nested
                {
                    rsu_replace(MED(cur, distance(orig, dest)));
                    out_buf = (short)M1;
                }
            }
            else if ((nH == 1 && typeH == 2) || (nH == 2 && typeH == 8) || (nH == 2 && typeH == 14)) // i-o family, exit / false-inside / virtual transitions
            {
                if (cur == 0) // false inside, IG, copy-through
                {
                }
                else if (buf == 0) // cavity exit, IG * S
                {
                    std::complex<double> S = MAT(cur).slab_airy_factor(MAT(0), MAT(prev_mat), theta, distance(orig, fbs), fGHz, eps, !prev_nonpar);
                    if (!std::isnan(std::real(S)))
                        rsu_scale(std::real(S), std::imag(S));
                    out_cur = (short)0;
                }
                else if (nH == 1 && typeH == 2) // virtual i-i
                {
                    if (SAME(buf, M1)) // M2 embedded in M1, ignore M2
                    {
                        rsu_replace(MED(cur, distance(orig, dest)));
                        out_buf = (short)0;
                    }
                    else
                    {
                        double g = MED(cur, distance(orig, fbs)) * TRN(cur, buf) * MED(buf, distance(fbs, dest));
                        rsu_replace(g);
                        out_cur = (short)buf;
                        out_buf = (short)0;
                    }
                }
                else // nH == 2 types 8/14, buf != 0: ii-oo
                {
                    double g = MED(cur, distance(orig, fbs)) * TRN(cur, 0);
                    rsu_replace(g);
                    out_cur = (short)0;
                    out_buf = (short)0;
                }
            }
            else if (nH == 2 && typeH == 1) // o-i-o
            {
                if (cur == 0) // IG (bare); current_out <- M1, +flag
                {
                    out_cur = (short)M1;
                    bool nonpar = fbs_sbs_not_parallel();
                    out_prev = (short)(nonpar ? (int)0x8000 : 0);
                }
                else // nested o-i-o
                {
                    rsu_replace(MED(cur, distance(orig, fbs)));
                    out_buf = (short)M1;
                }
            }
            else if (nH == 2 && typeH == 2) // i-o-i
            {
                if (buf == 0)
                {
                    if (M2 == 0) // illegal
                        rsu_kill();
                    else // cavity exit, air gap: slab is air, bounded by M1 / M2
                    {
                        std::complex<double> S = MAT(0).slab_airy_factor(MAT(M1), MAT(M2), theta, distance(orig, fbs), fGHz, eps, !prev_nonpar);
                        if (!std::isnan(std::real(S)))
                            rsu_scale(std::real(S), std::imag(S));
                        out_cur = (short)0; // survives
                    }
                }
                else if (cur != 0)
                {
                    if (SAME(buf, M1))
                    {
                        rsu_replace(MED(cur, distance(orig, fbs) + ray_offset));
                        out_buf = (short)0; // survives
                    }
                    else
                    {
                        double g = MED(cur, distance(orig, fbs)) * TRN(cur, buf) * MED(buf, ray_offset);
                        rsu_replace(g);
                        out_cur = (short)buf;
                        out_buf = (short)0; // survives
                    }
                }
                else // buf != 0 and cur == 0: terminate (source lines 649-650)
                    rsu_kill();
            }
            else if (nH == 2 && (typeH == 4 || typeH == 5)) // M2M (i-i) family
            {
                if (cur == 0) // illegal
                    rsu_kill();
                else if (buf == 0)
                {
                    if (M1 == 0 || M2 == 0) // illegal
                        rsu_kill();
                    else // cavity transition: IG * S * MED(iM, d(fbs,dest) - off (unclamped))
                    {
                        int iM = (typeH == 5) ? M2 : M1;
                        double gmed = MED(iM, distance(fbs, dest) - ray_offset); // unclamped
                        std::complex<double> S = MAT(cur).slab_airy_factor(MAT(iM), MAT(prev_mat), theta, distance(orig, fbs), fGHz, eps, !prev_nonpar);
                        double cr = std::sqrt(gmed), ci = 0.0;
                        out_cur = (short)iM; // current_out <- iM
                        if (!std::isnan(std::real(S)))
                        {
                            cr = std::real(S) * std::sqrt(gmed), ci = std::imag(S) * std::sqrt(gmed);
                            out_cur = (short)((iM & 0x7FFF) | (int)0x8000); // resolved: later crossings are transparent
                        }
                        rsu_scale(cr, ci);
                        out_prev = (short)cur; // prev_out <- old cur
                    }
                }
                else // buf != 0: ignore hit, continue in cur, swap buffer
                {
                    rsu_replace(MED(cur, distance(orig, dest)));
                    out_buf = (short)(SAME(buf, M1) ? M2 : M1);
                }
            }
            else if (nH == 2 && typeH == 10) // Edge o-i-o; No S (graze, not a slab).
            {
                if (cur == 0) // IG; current_out <- 0
                    out_cur = (short)0;
                else if (SAME(M1, M2)) // ignore hit
                    rsu_replace(MED(cur, distance(orig, dest)));
                else // i-i transition
                {
                    double g = MED(cur, distance(orig, fbs)) * TRN(cur, M1) * MED(M1, distance(fbs, dest));
                    rsu_replace(g);
                    out_cur = (short)M1;
                }
            }
            else if (nH == 2 && typeH == 11) // Edge i-o-i. No S, no flag (edge normals not a slab pair)
            {
                if (cur == 0) // IG; current_out <- (d(fbs,sbs) > 1e-6 ? M2 : 0)
                    out_cur = (short)((distance(fbs, sbs) > 1.0e-6) ? M2 : 0);
                else if (SAME(M1, M2)) // ignore hit
                    rsu_replace(MED(cur, distance(orig, dest)));
                else // i-i transition: MED(M2, d(fbs,dest) - off (unclamped))
                {
                    rsu_replace(MED(M2, distance(fbs, dest) - ray_offset));
                    out_cur = (short)M2;
                }
            }
            else if (nH > 2) // Multi-hit
            {
                if (cur == 0) // outside
                {
                    if (buf != 0) // cannot have i-i transition in buffer
                        rsu_kill();
                    else if (typeH == 1 || typeH == 7) // o-i; IG; current_out <- M1, +flag
                    {
                        out_cur = (short)M1;
                        bool nonpar = fbs_sbs_not_parallel();
                        out_prev = (short)(nonpar ? (int)0x8000 : 0);
                    }
                    else if (typeH == 2) // false inside: IG
                    {
                    }
                    else if (typeH == 10) // edge o-i-o, stay outside: IG
                    {
                    }
                    else if (typeH == 13) // edge o-i;  IG; current_out <- M1, buffer_out <- M2, +flag
                    {
                        out_cur = (short)M1;
                        out_buf = (short)M2;
                        bool nonpar = fbs_sbs_not_parallel();
                        out_prev = (short)(nonpar ? (int)0x8000 : 0);
                    }
                    else // some other hit type
                        rsu_kill();
                }
                else // inside
                {
                    if (typeH == 1 || typeH == 7 || typeH == 13) // nested o-i, overlapping mesh
                    {
                        rsu_replace(MED(cur, distance(orig, fbs) + ray_offset));
                        out_buf = (short)M1;
                    }
                    else if (typeH == 2 || typeH == 14) // i-o
                    {
                        if (buf == 0) // cavity exit, IG * S
                        {
                            std::complex<double> S = MAT(cur).slab_airy_factor(MAT(0), MAT(prev_mat), theta, distance(orig, fbs), fGHz, eps, !prev_nonpar);
                            if (!std::isnan(std::real(S)))
                                rsu_scale(std::real(S), std::imag(S));
                            out_cur = (short)0;
                        }
                        else if (SAME(buf, M1)) // M2 embedded in M1
                        {
                            rsu_replace(MED(cur, distance(orig, fbs) + ray_offset));
                            out_buf = (short)0;
                        }
                        else
                        {
                            double g = MED(cur, distance(orig, fbs)) * TRN(cur, buf) * MED(buf, ray_offset);
                            rsu_replace(g);
                            out_cur = (short)buf;
                            out_buf = (short)0;
                        }
                    }
                    else if (typeH == 4 || typeH == 5) // i-i
                    {
                        if (buf != 0) // spurious (probable false detection): IG
                            out_buf = (short)0;
                        else // cavity transition, IG * S
                        {
                            int iM = (typeH == 5) ? M2 : M1;
                            std::complex<double> S = MAT(cur).slab_airy_factor(MAT(iM), MAT(prev_mat), theta, distance(orig, fbs), fGHz, eps, !prev_nonpar);
                            out_cur = (short)iM; // current_out <- iM
                            if (!std::isnan(std::real(S)))
                            {
                                rsu_scale(std::real(S), std::imag(S));
                                out_cur = (short)((iM & 0x7FFF) | (int)0x8000); // resolved: later crossings are transparent
                            }
                            out_prev = (short)cur; // prev_out <- old cur
                        }
                    }
                    else if (typeH == 8) // overlapping i-o
                    {
                        if (buf == 0) // cavity exit, IG * S
                        {
                            std::complex<double> S = MAT(cur).slab_airy_factor(MAT(0), MAT(prev_mat), theta, distance(orig, fbs), fGHz, eps, !prev_nonpar);
                            if (!std::isnan(std::real(S)))
                                rsu_scale(std::real(S), std::imag(S));
                            out_cur = (short)0;
                        }
                        else
                        {
                            double g = MED(cur, distance(orig, fbs)) * TRN(cur, 0);
                            rsu_replace(g);
                            out_cur = (short)0;
                            out_buf = (short)0;
                        }
                    }
                    else if (typeH == 10) // edge o-i-o (the cur == 0 guard at source 824-828 is dead)
                    {
                        if (buf == 0)
                        {
                            if (SAME(M1, M2)) // ignore hit
                                rsu_replace(MED(cur, distance(orig, fbs) + ray_offset));
                            else // i-i transition
                            {
                                double g = MED(cur, distance(orig, fbs)) * TRN(cur, M1) * MED(M1, ray_offset);
                                rsu_replace(g);
                                out_cur = (short)M1;
                            }
                        }
                        else // buf != 0: virtual i-i
                        {
                            if (SAME(buf, M1))
                            {
                                rsu_replace(MED(cur, distance(orig, fbs) + ray_offset));
                                out_buf = (short)0;
                            }
                            else
                            {
                                double g = MED(cur, distance(orig, fbs)) * TRN(cur, buf) * MED(buf, ray_offset);
                                rsu_replace(g);
                                out_cur = (short)buf;
                                out_buf = (short)0;
                            }
                        }
                    }
                    else if (typeH == 11) // edge i-o-i
                    {
                        if (buf == 0)
                        {
                            if (SAME(M1, M2)) // ignore hit
                                rsu_replace(MED(cur, distance(orig, fbs) + ray_offset));
                            else // i-i transition
                            {
                                double g = MED(cur, distance(orig, fbs)) * TRN(cur, M2) * MED(M2, ray_offset);
                                rsu_replace(g);
                                out_cur = (short)M2;
                            }
                        }
                        else // buf != 0: spurious, IG
                        {
                            out_buf = (short)0;
                        }
                    }
                    else // Unmatched inside type (TR or a degenerate out_type 0)
                        rsu_kill();
                }
            }
            else // Global default: any unmatched (out_type, nH, state): KILL
                rsu_kill();
        }

        // Write the new state words (compact set)
        if (p_prev_out != nullptr)
            p_prev_out[ii] = out_prev;
        if (p_cur_out != nullptr)
            p_cur_out[ii] = out_cur;
        if (p_buf_out != nullptr)
            p_buf_out[ii] = out_buf;
    }
}

template void quadriga_lib::ray_state_update(int interaction_type, float center_frequency,
                                             const arma::Mat<float> *orig, const arma::Mat<float> *dest,
                                             const arma::Mat<float> *fbs, const arma::Mat<float> *sbs,
                                             const arma::u32_vec *no_interact, const arma::Col<float> *fbs_angleN,
                                             const arma::s32_vec *out_typeN,
                                             const std::unordered_map<std::string, std::vector<float>> *mtl_prop,
                                             const arma::Col<short> *mtl_ind_fbs, const arma::Col<short> *mtl_ind_sbs,
                                             const arma::Col<short> *mtl_ind_prev_in, const arma::Col<short> *mtl_ind_current_in,
                                             const arma::Col<short> *mtl_ind_buffer_in,
                                             const arma::Mat<float> *normal_vecN,
                                             arma::Col<short> *mtl_ind_prev_out, arma::Col<short> *mtl_ind_current_out,
                                             arma::Col<short> *mtl_ind_buffer_out,
                                             arma::Col<float> *gainN, arma::Mat<float> *xprmatN, arma::u32_vec *ray_ind, double eps);

template void quadriga_lib::ray_state_update(int interaction_type, double center_frequency,
                                             const arma::Mat<double> *orig, const arma::Mat<double> *dest,
                                             const arma::Mat<double> *fbs, const arma::Mat<double> *sbs,
                                             const arma::u32_vec *no_interact, const arma::Col<double> *fbs_angleN,
                                             const arma::s32_vec *out_typeN,
                                             const std::unordered_map<std::string, std::vector<double>> *mtl_prop,
                                             const arma::Col<short> *mtl_ind_fbs, const arma::Col<short> *mtl_ind_sbs,
                                             const arma::Col<short> *mtl_ind_prev_in, const arma::Col<short> *mtl_ind_current_in,
                                             const arma::Col<short> *mtl_ind_buffer_in,
                                             const arma::Mat<double> *normal_vecN,
                                             arma::Col<short> *mtl_ind_prev_out, arma::Col<short> *mtl_ind_current_out,
                                             arma::Col<short> *mtl_ind_buffer_out,
                                             arma::Col<double> *gainN, arma::Mat<double> *xprmatN, arma::u32_vec *ray_ind, double eps);