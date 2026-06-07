// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// A collection of small reusable helper functions to reduce copy and pasting code

#ifndef quadriga_lib_material_H
#define quadriga_lib_material_H

#include <armadillo>
#include <cstring>
#include <complex>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>

// Validate a named material-property map and return the material count (n_mtl).
// - Every present (non-empty) column must have the same length; throws otherwise.
// - Empty columns are treated as absent (consumers apply per-column defaults).
// - An empty map returns 0.
// Call once per public entry point; internal mtl_col / mtl_val accesses are then safe
// for any material index < the returned n_mtl.
template <typename dtype>
static inline arma::uword mtl_validate(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop)
{
    arma::uword n_mtl = 0;
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
    return n_mtl;
}

// Resolve a named material-property column to a raw pointer (map-based material model).
// Returns nullptr if the column is absent; the consumer then applies its own default.
template <typename dtype>
static inline const dtype *mtl_col(const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop, const std::string &key)
{
    auto it = mtl_prop->find(key);
    return (it == mtl_prop->end() || it->second.empty()) ? nullptr : it->second.data();
}

// Read a material value for material index iM from a resolved column pointer.
// Falls back to 'def' when the column is absent.
template <typename dtype>
static inline double mtl_val(const dtype *col, arma::uword iM, double def)
{
    return (col == nullptr) ? def : (double)col[iM];
}

// Assemble complex-valued eta from mtl coefficients
static inline std::complex<double> eta_from_coeffs(double a, double b, double c, double d, double fRef, double fGHz)
{
    if (fRef <= 0.0)
        fRef = 1.0;
    double f_rel = fGHz / fRef;
    double eta_r = a * std::pow(f_rel, b);
    double sigma = c * std::pow(f_rel, d);
    double eta_i = -17.98 * sigma / fGHz;
    return std::complex<double>(eta_r, eta_i);
}

// Relative permeability from mtl coefficients (mu = 1 when columns absent -> EM unchanged)
static inline std::complex<double> mu_from_coeffs(double e, double f, double g, double h, double fRef, double fGHz)
{
    if (fRef <= 0.0)
        fRef = 1.0;
    double f_rel = fGHz / fRef;
    double mu_r = e * std::pow(f_rel, f);
    double sigma_m = g * std::pow(f_rel, h);
    return std::complex<double>(mu_r, -17.98 * sigma_m / fGHz);
}

// Permittivity resonance (acoustic): complex Lorentz pole added to the interface (Fresnel)
// permittivity only. Inactive unless resF > 0, resQ > 0 and resS != 0, so the EM path is
// unchanged. The +i denominator makes resS > 0 add loss (negative imaginary part), consistent
// with the conductivity term. Deliberately NOT applied to the in-medium loss: a strong pole can
// push Re(eta) < 0, and medium_loss_dB uses a real sqrt(Re eta).
static inline std::complex<double> eta_resonance(double resF, double resQ, double resS, double fGHz)
{
    if (resF <= 0.0 || resQ <= 0.0 || resS == 0.0)
        return std::complex<double>(0.0, 0.0);
    double resF2 = resF * resF;
    std::complex<double> denom(resF2 - fGHz * fGHz, (resF / resQ) * fGHz);
    return (resS * resF2) / denom;
}

// Calculate in-medium loss
static inline double medium_loss_dB(std::complex<double> eta, double alpha, double alphaB,
                                    double fRef, double fGHz, double dist, double mass = 0.0,
                                    double abs_cos_theta = 1.0)
{
    if (fRef <= 0.0)
        fRef = 1.0;
    double er = std::real(eta);
    double tan_delta = std::imag(eta) / er;
    double cos_delta = 1.0 / std::sqrt(1.0 + tan_delta * tan_delta);
    double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
    Delta = std::sqrt(Delta) * 0.0477135 / (fGHz * std::sqrt(er));
    double loss = dist * 8.686 / Delta;
    loss += dist * alpha * std::pow(fGHz / fRef, alphaB);

    // Mass is a bulk-propagation term: never apply it over the ~1 mm co-location
    // epsilon (ray_offset). Real traversals are at least the panel thickness (cm),
    // so a small path floor removes the spurious slope without touching them.
    constexpr double mass_min_path = 0.0015; // m, above ray_offset (0.001)
    if (mass > 0.0 && dist > mass_min_path)
    {
        // Mass law is a surface-impedance term: its angle factor is cos(theta), not the
        // 1/cos(theta) of the slant traversal. dist is the slant path d/cos(theta), so
        // dist * cos^2(theta) = d * cos(theta) recovers the mass-law surface mass.
        double mass_path = dist * abs_cos_theta * abs_cos_theta;
        double m_dB = mass * std::log10((fGHz / fRef) * mass_path);
        if (m_dB > 0.0)
            loss += m_dB;
    }
    return loss;
}

// In-medium gain for material index iM. No validation: the caller guarantees a
// column-consistent map (via mtl_validate / obj_file_read) and iM < n_mtl.
template <typename dtype>
static inline dtype medium_gain_impl(const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop, arma::uword iM,
                                     dtype dist, dtype center_frequency)
{
    if (!mtl_prop)
        return (dtype)1.0;
    double fGHz = (double)center_frequency * 1e-9;
    const dtype *m_a = mtl_col(mtl_prop, "a");
    const dtype *m_b = mtl_col(mtl_prop, "b");
    const dtype *m_c = mtl_col(mtl_prop, "c");
    const dtype *m_d = mtl_col(mtl_prop, "d");
    const dtype *m_e = mtl_col(mtl_prop, "e");
    const dtype *m_f = mtl_col(mtl_prop, "f");
    const dtype *m_g = mtl_col(mtl_prop, "g");
    const dtype *m_h = mtl_col(mtl_prop, "h");
    const dtype *m_alpha = mtl_col(mtl_prop, "alpha");
    const dtype *m_alphaB = mtl_col(mtl_prop, "alphaB");
    const dtype *m_mass = mtl_col(mtl_prop, "m");
    const dtype *m_fRef = mtl_col(mtl_prop, "fRef");

    std::complex<double> eta = eta_from_coeffs(mtl_val(m_a, iM, 1.0), mtl_val(m_b, iM, 0.0),
                                               mtl_val(m_c, iM, 0.0), mtl_val(m_d, iM, 0.0),
                                               mtl_val(m_fRef, iM, 1.0), fGHz);

    std::complex<double> mu = mu_from_coeffs(mtl_val(m_e, iM, 1.0), mtl_val(m_f, iM, 0.0),
                                             mtl_val(m_g, iM, 0.0), mtl_val(m_h, iM, 0.0),
                                             mtl_val(m_fRef, iM, 1.0), fGHz);

    double A = medium_loss_dB(eta * mu, mtl_val(m_alpha, iM, 0.0), mtl_val(m_alphaB, iM, 0.0),
                              mtl_val(m_fRef, iM, 1.0), fGHz, (double)dist, mtl_val(m_mass, iM, 0.0));

    return (dtype)std::pow(10.0, -0.1 * A);
}

// Lumped per-entry interface attenuation in dB: power-law penetration loss plus an optional
// Lorentzian coincidence feature. Coincidence is active only when coiF > 0 and coiA != 0.
// Total is clamped to >= 0 (a coincidence dip cannot create transmission gain).
static inline double interface_loss_dB(double att, double attB,
                                       double coiF, double coiQ, double coiA,
                                       double fRef, double fGHz)
{
    if (fRef <= 0.0)
        fRef = 1.0;
    double loss = att * std::pow(fGHz / fRef, attB);
    if (coiF > 0.0 && coiA != 0.0)
    {
        double x = coiQ * (fGHz - coiF) / coiF;
        loss += coiA / (1.0 + x * x);
    }
    return loss;
}

// Lumped interface transmission gain for material index iM (the material being entered).
// No validation: the caller guarantees a column-consistent map (via mtl_validate /
// obj_file_read) and iM < n_mtl. Path-independent; applied once on entry.
template <typename dtype>
static inline dtype interface_gain_impl(const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop, arma::uword iM, dtype center_frequency)
{
    if (!mtl_prop)
        return (dtype)1.0;

    double fGHz = (double)center_frequency * 1e-9;
    const dtype *m_att = mtl_col(mtl_prop, "att");
    const dtype *m_attB = mtl_col(mtl_prop, "attB");
    const dtype *m_coiF = mtl_col(mtl_prop, "coiF");
    const dtype *m_coiQ = mtl_col(mtl_prop, "coiQ");
    const dtype *m_coiA = mtl_col(mtl_prop, "coiA");
    const dtype *m_fRef = mtl_col(mtl_prop, "fRef");

    double A = interface_loss_dB(mtl_val(m_att, iM, 0.0), mtl_val(m_attB, iM, 0.0),
                                 mtl_val(m_coiF, iM, 0.0), mtl_val(m_coiQ, iM, 0.0),
                                 mtl_val(m_coiA, iM, 0.0), mtl_val(m_fRef, iM, 1.0), fGHz);
    return (dtype)std::pow(10.0, -0.1 * A);
}

// Transmission factor at fGHz with optional power-law slope tfB about fRef, clamped to [-1, 1].
// Redistributes energy between the reflected and transmitted paths (see tf_apply).
static inline double tf_value(double tf, double tfB, double fRef, double fGHz)
{
    if (fRef <= 0.0)
        fRef = 1.0;
    double v = tf * std::pow(fGHz / fRef, tfB);
    return (v < -1.0) ? -1.0 : ((v > 1.0) ? 1.0 : v);
}

// Redistribute physical reflection energy R0 in [0,1] by tf in [-1,1], keeping refl + trans = 1.
// tf = 0 -> R0 (physical Fresnel); tf = +1 -> 0 (fully transparent); tf = -1 -> 1 (fully reflective).
static inline double tf_apply(double R0, double tf)
{
    R0 = (R0 < 0.0) ? 0.0 : ((R0 > 1.0) ? 1.0 : R0); // guard against resonance overshoot
    return (tf >= 0.0) ? R0 * (1.0 - tf) : R0 + (1.0 - R0) * (-tf);
}

#endif