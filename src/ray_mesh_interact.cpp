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

// Material-interaction helpers. Formerly in "quadriga_lib_material_helpers.hpp"; merged here so
// that ray_mesh_interact and ray_state_update share one translation unit (the include is dropped).

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
- **`iM`** — 0-based material index selecting the material from `mtl_prop`
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
    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    arma::uword n_mtl = mtl_validate(mtl_prop);
    if (iM >= n_mtl)
        throw std::invalid_argument("Material index out of bound.");
    return medium_gain_impl(&mtl_prop, iM, dist, center_frequency);
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
- **`iM`** — 0-based material index selecting the entered material from `mtl_prop`
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
    arma::uword n_mtl = mtl_validate(mtl_prop);
    if (iM >= n_mtl)
        throw std::invalid_argument("Material index out of bound.");
    return interface_gain_impl(&mtl_prop, iM, center_frequency);
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
- **`mtl_ind`** — 0-based material index per face (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`. NULL → all faces treated as air.
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`. NULL → air defaults used.
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian
- **`orig_length`** *(optional)* — Accumulated path length at origin; `[n_ray]`, default 0

## Outputs:
- **`origN`** — New origins after interaction (offset 0.001 m along travel direction); `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`gainN`** — Interaction gain (linear scale, includes in-medium attenuation, excludes FSPL); averaged over TE/TM polarizations for types 0–2, TE-only for types 3–4; `[n_rayN]`
- **`xprmatN`** — For types 0–2: polarization transfer matrix, interleaved complex `[ReVV ImVV ReVH ImVH ReHV ImHV ReHH ImHH]`; includes interaction gain, TE/TM coefficients, incidence plane orientation, in-medium attenuation (excludes FSPL); `[n_rayN, 8]`. For types 3–4 (scalar): `[Re Im 0 0 0 0 0 0]` where Re+jIm is the scalar pressure coefficient including in-medium attenuation; `[n_rayN, 8]`.
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
- [[obj_file_read]] (for loading `mesh` and `mtl_prop` from OBJ file)
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

    const arma::uword n_ray = orig->n_rows;     // Number of rays
    const arma::uword n_mesh = mesh->n_rows;    // Number of mesh elements
    const int n_ray_i = (int)n_ray;             // Number of rays as int
    const unsigned n_mesh_u = (unsigned)n_mesh; // Number of mesh elements as unsigned int
    const size_t n_ray_t = (size_t)n_ray;       // Number of rays as size_t
    const size_t n_mesh_t = (size_t)n_mesh;     // Number of mesh elements as size_t

    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

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
    const dtype *p_trivec = (trivec == nullptr) ? nullptr : trivec->memptr();
    const dtype *p_tridir = (tridir == nullptr) ? nullptr : tridir->memptr();
    const dtype *p_orig_length = (orig_length == nullptr) ? nullptr : orig_length->memptr();

    // Resolve material columns once; nullptr -> per-column default applied in the ray loop
    const arma::uword *p_mtl_ind = (mtl_ind == nullptr || mtl_ind->is_empty()) ? nullptr : mtl_ind->memptr();
    const dtype *m_a = nullptr, *m_b = nullptr, *m_c = nullptr, *m_d = nullptr;
    const dtype *m_e = nullptr, *m_f = nullptr, *m_g = nullptr, *m_h = nullptr;
    const dtype *m_alpha = nullptr, *m_alphaB = nullptr, *m_mass = nullptr, *m_fRef = nullptr;
    const dtype *m_resF = nullptr, *m_resQ = nullptr, *m_resS = nullptr;
    const dtype *m_tf = nullptr, *m_tfB = nullptr;
    arma::uword n_mtl = 0;
    if (mtl_prop != nullptr)
    {
        n_mtl = mtl_validate(*mtl_prop); // all columns share length n_mtl; throws on mismatch
        m_a = mtl_col(mtl_prop, "a");
        m_b = mtl_col(mtl_prop, "b");
        m_c = mtl_col(mtl_prop, "c");
        m_d = mtl_col(mtl_prop, "d");
        m_e = mtl_col(mtl_prop, "e");
        m_f = mtl_col(mtl_prop, "f");
        m_g = mtl_col(mtl_prop, "g");
        m_h = mtl_col(mtl_prop, "h");
        m_alpha = mtl_col(mtl_prop, "alpha");
        m_alphaB = mtl_col(mtl_prop, "alphaB");
        m_mass = mtl_col(mtl_prop, "m");
        m_fRef = mtl_col(mtl_prop, "fRef");
        m_resF = mtl_col(mtl_prop, "resF");
        m_resQ = mtl_col(mtl_prop, "resQ");
        m_resS = mtl_col(mtl_prop, "resS");
        m_tf = mtl_col(mtl_prop, "tf");
        m_tfB = mtl_col(mtl_prop, "tfB");
    }
    if (p_mtl_ind != nullptr && n_mtl > 0 && mtl_ind->max() >= n_mtl)
        throw std::invalid_argument("Values in 'mtl_ind' exceed the number of materials in 'mtl_prop'.");

    // Get number of output rays and build output ray index
    // - Only consider rays that have at least one interaction with the mesh, i.e. 'fbs_ind != 0'
    unsigned n_rayN_u = 0;
    unsigned *output_ray_index = new unsigned[n_ray_t]; // 1-based
    for (size_t i_ray = 0; i_ray < n_ray_t; ++i_ray)    // Ray loop
        if (p_fbs_ind[i_ray] == 0)                      // No hit
            output_ray_index[i_ray] = 0;
        else if (p_fbs_ind[i_ray] > n_mesh_u) // Invalid, must be 1 ... n_mesh (1-based index)
            throw std::invalid_argument("Some values in 'fbs_ind' exceed number of mesh elements.");
        else // Store value
            output_ray_index[i_ray] = ++n_rayN_u;

    const arma::uword n_rayN = (arma::uword)n_rayN_u;
    const size_t n_rayN_t = (size_t)n_rayN_u;

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
    for (int i_ray = 0; i_ray < n_ray_i; ++i_ray) // Ray loop
    {
        if (p_fbs_ind[i_ray] == 0) // Skip non-hits
            continue;

        size_t iRx = (size_t)i_ray;                 // Ray x-index
        size_t iRy = iRx + n_ray_t;                 // Ray y-index
        size_t iRz = iRy + n_ray_t;                 // Ray z-index
        size_t iFBS = (size_t)p_fbs_ind[i_ray] - 1; // Mesh FBS index, 0-based

        // SBS index
        size_t iSBS = (size_t)sbs_ind->at(iRx); // Mesh SBS index, 1-based
        if (iSBS > n_mesh_t)
            throw std::invalid_argument("Some values in 'sbs_ind' exceed number of mesh elements.");

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
               V1y = (double)p_mesh[iFBS + n_mesh_t],
               V1z = (double)p_mesh[iFBS + 2 * n_mesh_t];
        double E1x = (double)p_mesh[iFBS + 3 * n_mesh_t] - V1x,
               E1y = (double)p_mesh[iFBS + 4 * n_mesh_t] - V1y,
               E1z = (double)p_mesh[iFBS + 5 * n_mesh_t] - V1z;
        double E2x = (double)p_mesh[iFBS + 6 * n_mesh_t] - V1x,
               E2y = (double)p_mesh[iFBS + 7 * n_mesh_t] - V1y,
               E2z = (double)p_mesh[iFBS + 8 * n_mesh_t] - V1z;
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
            V1y = (double)p_mesh[iSBS - 1 + n_mesh_t],
            V1z = (double)p_mesh[iSBS - 1 + 2 * n_mesh_t];
            E1x = (double)p_mesh[iSBS - 1 + 3 * n_mesh_t] - V1x,
            E1y = (double)p_mesh[iSBS - 1 + 4 * n_mesh_t] - V1y,
            E1z = (double)p_mesh[iSBS - 1 + 5 * n_mesh_t] - V1z;
            E2x = (double)p_mesh[iSBS - 1 + 6 * n_mesh_t] - V1x,
            E2y = (double)p_mesh[iSBS - 1 + 7 * n_mesh_t] - V1y,
            E2z = (double)p_mesh[iSBS - 1 + 8 * n_mesh_t] - V1z;
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

        // Select the properties of the two materials
        double kR1 = 1.0, kR2 = 0.0, kR3 = 0.0, kR4 = 0.0;     // medium 1: a, b, c, d
        double kR5 = 1.0, kR6 = 0.0, kR7 = 0.0, kR8 = 0.0;     // medium 1: e, f, g, h (mu)
        double kR_alpha = 0.0, kR_alphaB = 0.0, kR_fRef = 1.0; // medium 1: alpha, alphaB, fRef
        double kS1 = 1.0, kS2 = 0.0, kS3 = 0.0, kS4 = 0.0;     // medium 2: a, b, c, d
        double kS5 = 1.0, kS6 = 0.0, kS7 = 0.0, kS8 = 0.0;     // medium 2: e, f, g, h (mu)
        double kS_alpha = 0.0, kS_alphaB = 0.0, kS_fRef = 1.0; // medium 2: alpha, alphaB, fRef
        double kR_mass = 0.0, kS_mass = 0.0;
        double kR_resF = 0.0, kR_resQ = 0.0, kR_resS = 0.0;
        double kS_resF = 0.0, kS_resQ = 0.0, kS_resS = 0.0;
        double kR_tf = 0.0, kR_tfB = 0.0, kS_tf = 0.0, kS_tfB = 0.0;
        double transition_gain = 1.0;

        if (theta >= 0.0) // Ray hits front side of FBS/SBS face, set second material to object material
        {
            kS1 = mtl_val(m_a, iMF, 1.0);
            kS2 = mtl_val(m_b, iMF, 0.0);
            kS3 = mtl_val(m_c, iMF, 0.0);
            kS4 = mtl_val(m_d, iMF, 0.0);
            kS5 = mtl_val(m_e, iMF, 1.0);
            kS6 = mtl_val(m_f, iMF, 0.0);
            kS7 = mtl_val(m_g, iMF, 0.0);
            kS8 = mtl_val(m_h, iMF, 0.0);
            kS_alpha = mtl_val(m_alpha, iMF, 0.0);
            kS_alphaB = mtl_val(m_alphaB, iMF, 0.0);
            kS_mass = mtl_val(m_mass, iMF, 0.0);
            kS_fRef = mtl_val(m_fRef, iMF, 1.0);
            kS_resF = mtl_val(m_resF, iMF, 0.0);
            kS_resQ = mtl_val(m_resQ, iMF, 0.0);
            kS_resS = mtl_val(m_resS, iMF, 0.0);
            kS_tf = mtl_val(m_tf, iMF, 0.0);
            kS_tfB = mtl_val(m_tfB, iMF, 0.0);
            transition_gain = (double)interface_gain_impl(mtl_prop, iMF, center_frequency);
        }
        else // Ray hits back side of FBS face, set first material to object material
        {
            kR1 = mtl_val(m_a, iMF, 1.0);
            kR2 = mtl_val(m_b, iMF, 0.0);
            kR3 = mtl_val(m_c, iMF, 0.0);
            kR4 = mtl_val(m_d, iMF, 0.0);
            kR5 = mtl_val(m_e, iMF, 1.0);
            kR6 = mtl_val(m_f, iMF, 0.0);
            kR7 = mtl_val(m_g, iMF, 0.0);
            kR8 = mtl_val(m_h, iMF, 0.0);
            kR_alpha = mtl_val(m_alpha, iMF, 0.0);
            kR_alphaB = mtl_val(m_alphaB, iMF, 0.0);
            kR_mass = mtl_val(m_mass, iMF, 0.0);
            kR_fRef = mtl_val(m_fRef, iMF, 1.0);
            kR_resF = mtl_val(m_resF, iMF, 0.0);
            kR_resQ = mtl_val(m_resQ, iMF, 0.0);
            kR_resS = mtl_val(m_resS, iMF, 0.0);
            kR_tf = mtl_val(m_tf, iMF, 0.0);
            kR_tfB = mtl_val(m_tfB, iMF, 0.0);
        }

        if (material_to_material) // Material to material transition
        {
            if (theta >= 0.0) // SBS (front side) is hit first
            {
                kR1 = mtl_val(m_a, iMS, 1.0);
                kR2 = mtl_val(m_b, iMS, 0.0);
                kR3 = mtl_val(m_c, iMS, 0.0);
                kR4 = mtl_val(m_d, iMS, 0.0);
                kR5 = mtl_val(m_e, iMS, 1.0);
                kR6 = mtl_val(m_f, iMS, 0.0);
                kR7 = mtl_val(m_g, iMS, 0.0);
                kR8 = mtl_val(m_h, iMS, 0.0);
                kR_alpha = mtl_val(m_alpha, iMS, 0.0);
                kR_alphaB = mtl_val(m_alphaB, iMS, 0.0);
                kR_mass = mtl_val(m_mass, iMS, 0.0);
                kR_fRef = mtl_val(m_fRef, iMS, 1.0);
                kR_resF = mtl_val(m_resF, iMS, 0.0);
                kR_resQ = mtl_val(m_resQ, iMS, 0.0);
                kR_resS = mtl_val(m_resS, iMS, 0.0);
                kR_tf = mtl_val(m_tf, iMS, 0.0);
                kR_tfB = mtl_val(m_tfB, iMS, 0.0);
            }
            else // FBS (back side) is hit first
            {
                kS1 = mtl_val(m_a, iMS, 1.0);
                kS2 = mtl_val(m_b, iMS, 0.0);
                kS3 = mtl_val(m_c, iMS, 0.0);
                kS4 = mtl_val(m_d, iMS, 0.0);
                kS5 = mtl_val(m_e, iMS, 1.0);
                kS6 = mtl_val(m_f, iMS, 0.0);
                kS7 = mtl_val(m_g, iMS, 0.0);
                kS8 = mtl_val(m_h, iMS, 0.0);
                kS_alpha = mtl_val(m_alpha, iMS, 0.0);
                kS_alphaB = mtl_val(m_alphaB, iMS, 0.0);
                kS_mass = mtl_val(m_mass, iMS, 0.0);
                kS_fRef = mtl_val(m_fRef, iMS, 1.0);
                kS_resF = mtl_val(m_resF, iMS, 0.0);
                kS_resQ = mtl_val(m_resQ, iMS, 0.0);
                kS_resS = mtl_val(m_resS, iMS, 0.0);
                kS_tf = mtl_val(m_tf, iMS, 0.0);
                kS_tfB = mtl_val(m_tfB, iMS, 0.0);
                transition_gain = (double)interface_gain_impl(mtl_prop, iMS, center_frequency);
            }
        }

        // Complex-valued relative permittivity, ITU-R P.2040-1 eq. (9b)
        std::complex<double> eta1 = eta_from_coeffs(kR1, kR2, kR3, kR4, kR_fRef, fGHz);
        std::complex<double> eta2 = eta_from_coeffs(kS1, kS2, kS3, kS4, kS_fRef, fGHz);
        std::complex<double> mu1 = mu_from_coeffs(kR5, kR6, kR7, kR8, kR_fRef, fGHz);
        std::complex<double> mu2 = mu_from_coeffs(kS5, kS6, kS7, kS8, kS_fRef, fGHz);

        // base permittivity for the in-medium loss (no resonance: keeps the real-sqrt loss well-posed)
        std::complex<double> eta1_med = eta1, eta2_med = eta2;
        // resonance enters the interface permittivity only
        eta1 += eta_resonance(kR_resF, kR_resQ, kR_resS, fGHz);
        eta2 += eta_resonance(kS_resF, kS_resQ, kS_resS, fGHz);

        bool dense2light = std::real(eta1 * mu1) > std::real(eta2 * mu2);

        // Evaluate total reflection condition in ITU-R P.2040-1, eq. (31) and (32)
        double sin_theta = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta);  // Trigonometric identity
        std::complex<double> eta1_div_eta2 = (eta1 * mu1) / (eta2 * mu2);   // (n1/n2)^2 = eps*mu ratio
        double eta = std::sqrt(std::abs(eta1_div_eta2));                    // sgrt( abs( eta1 / eta2 ) )
        bool total_reflection = is_scalar ? false : eta * sin_theta >= 1.0; // Total reflection condition

        // Calculate cos_theta2 from Rec. ITU-R P.2040-1, eq. (33)
        std::complex<double> cos_theta2 = std::sqrt(1.0 - eta1_div_eta2 * sin_theta * sin_theta);

        // Calculate the center path direction after medium interaction (normalized to length 1)
        double FDx = Dx - Fx, FDy = Dy - Fy, FDz = Dz - Fz;              // Vector from FBS to destination
        double FD_length = std::sqrt(FDx * FDx + FDy * FDy + FDz * FDz); // Length of path from FBS to destination

        if (geometry_type == 0) // Reflection, normalized by default
            FDx = OFx - 2.0 * cos_theta * Nx,
            FDy = OFy - 2.0 * cos_theta * Ny,
            FDz = OFz - 2.0 * cos_theta * Nz;
        else if (geometry_type == 1)         // Transmission without refraction
            FDx = OFx, FDy = OFy, FDz = OFz; // New path direction = same as incoming ray, already normalized
        else                                 // Refraction
        {
            scl = eta * abs_cos_theta - std::real(cos_theta2);                                            // Temporary variable, ignoring imaginary part of cos_theta2
            FDx = eta * OFx + scl * Nx, FDy = eta * OFy + scl * Ny, FDz = eta * OFz + scl * Nz;           // Refraction into medium
            scl = 1.0 / std::sqrt(FDx * FDx + FDy * FDy + FDz * FDz), FDx *= scl, FDy *= scl, FDz *= scl; // Normalize
        }

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
                    Tx += (double)p_trivec[iRx + 3 * n_ray_t], Ty += (double)p_trivec[iRx + 4 * n_ray_t], Tz += (double)p_trivec[iRx + 5 * n_ray_t];
                    if (use_ray_tube == 1)
                        az = (double)p_tridir[iRx + 2 * n_ray_t], el = (double)p_tridir[iRx + 3 * n_ray_t];
                    else
                        Vx = (double)p_tridir[iRx + 3 * n_ray_t], Vy = (double)p_tridir[iRx + 4 * n_ray_t], Vz = (double)p_tridir[iRx + 5 * n_ray_t];
                }
                else if (iTube == 3)
                {
                    Tx += (double)p_trivec[iRx + 6 * n_ray_t], Ty += (double)p_trivec[iRx + 7 * n_ray_t], Tz += (double)p_trivec[iRx + 8 * n_ray_t];
                    if (use_ray_tube == 1)
                        az = (double)p_tridir[iRx + 4 * n_ray_t], el = (double)p_tridir[iRx + 5 * n_ray_t];
                    else
                        Vx = (double)p_tridir[iRx + 6 * n_ray_t], Vy = (double)p_tridir[iRx + 7 * n_ray_t], Vz = (double)p_tridir[iRx + 8 * n_ray_t];
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
                    double a = 2.0 * (Vx * Nx + Vy * Ny + Vz * Nz);
                    Vx -= a * Nx, Vy -= a * Ny, Vz -= a * Nz;
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
                        double sin_thetaV = std::sqrt(1.0 - cos_thetaV * cos_thetaV);    // Sine of incidence angle
                        total_reflection = total_reflection | (eta * sin_thetaV >= 1.0); // Check total reflection condition

                        // Refraction into medium
                        std::complex<double> cos_theta2V = std::sqrt(1.0 - eta1_div_eta2 * sin_thetaV * sin_thetaV);
                        double scl = eta * cos_thetaV - std::real(cos_theta2V);
                        Vx = eta * Vx + scl * Nx, Vy = eta * Vy + scl * Ny, Vz = eta * Vz + scl * Nz;

                        scl = 1.0 / std::sqrt(Vx * Vx + Vy * Vy + Vz * Vz);
                        Vx *= scl, Vy *= scl, Vz *= scl; // Normalize
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
        double gain = 1.0;
        if (ray_starts_inside)
        {
            double thickness = (geometry_type == 0) ? OF_length + ray_offset : OF_length;
            double loss_dB = medium_loss_dB(eta1_med * mu1, kR_alpha, kR_alphaB, kR_fRef, fGHz, thickness, kR_mass, abs_cos_theta);
            gain *= std::pow(10.0, -0.1 * loss_dB);
        }
        if (geometry_type != 0)
        {
            double loss_dB = medium_loss_dB(eta2_med * mu2, kS_alpha, kS_alphaB, kS_fRef, fGHz, ray_offset, kS_mass);
            gain *= std::pow(10.0, -0.1 * loss_dB);
        }

        // Add additional transition gain
        if (geometry_type != 0)
            gain *= transition_gain;

        // Calculate sqrt(eta1) and sqrt(eta2) needed for ITU-R P.2040-1, eq. (31) and (32)
        eta1 = std::sqrt(eta1 / mu1); // TE/scalar admittance sqrt(eps/mu)
        eta2 = std::sqrt(eta2 / mu2);

        // Calculate Reflection coefficients  ITU-R P.2040-1, eq. (31)
        std::complex<double> R_eTE = total_reflection ? 1.0 : 0.0;
        std::complex<double> R_eTM = total_reflection ? 1.0 : 0.0;
        double reflection_gain = total_reflection ? 1.0 : 0.0;

        if (is_scalar)
        {
            R_eTE = (eta1 * abs_cos_theta - eta2 * cos_theta2) / (eta1 * abs_cos_theta + eta2 * cos_theta2);
            R_eTM = R_eTE;                      // not used, but keeps downstream code safe
            reflection_gain = std::norm(R_eTE); // no 0.5 factor
        }
        else if (interaction_type == 1 || (interaction_type == 0 && !total_reflection)) // Reflection and Transmission
        {
            R_eTE = (eta1 * abs_cos_theta - eta2 * cos_theta2) / (eta1 * abs_cos_theta + eta2 * cos_theta2);
            R_eTM = (eta2 * abs_cos_theta - eta1 * cos_theta2) / (eta2 * abs_cos_theta + eta1 * cos_theta2);
            reflection_gain = 0.5 * (std::norm(R_eTE) + std::norm(R_eTM));
        }

        // Calculate Transmission coefficients  ITU-R P.2040-1, eq. (32)
        std::complex<double> T_eTE(0.0, 0.0), T_eTM(0.0, 0.0);
        double refraction_gain = 0.0;

        if (!total_reflection && !is_scalar && interaction_type != 0) // Transmission and Refraction
        {
            T_eTE = (2.0 * eta1 * abs_cos_theta) / (eta1 * abs_cos_theta + eta2 * cos_theta2);
            T_eTM = (2.0 * eta1 * abs_cos_theta) / (eta2 * abs_cos_theta + eta1 * cos_theta2);
            refraction_gain = 0.5 * (std::norm(T_eTE) + std::norm(T_eTM));
        }

        // Scalar transmission factor: redistribute reflection/transmission energy keeping their
        // sum at 1 (always conserved). tf = 0 leaves the physical Fresnel split; tf > 0 leaks
        // reflected energy into transmission, tf < 0 the reverse. tf_eff is the FBS face material's
        // factor (kS at front, kR at back) so it is independent of entry/exit segment placement.
        if (is_scalar)
        {
            double tf_eff = tf_value((theta >= 0.0) ? kS_tf : kR_tf,
                                     (theta >= 0.0) ? kS_tfB : kR_tfB,
                                     (theta >= 0.0) ? kS_fRef : kR_fRef, fGHz);
            double R0 = reflection_gain;        // physical |R_TE|^2 (== 1 when cos_theta2 is imaginary)
            double refl = tf_apply(R0, tf_eff); // redistributed reflection energy in [0,1]

            // Capture phases from the physical coefficient before rescaling. For TE, t = 1 + r
            // exactly, so the transmission phase is arg(1 + R_eTE). This stays well-defined under
            // total reflection (|R| == 1, physical T magnitude == 0), where arg(T_eTE) would read
            // arg(0) == 0 and silently drop the evanescent phase, breaking multi-bounce coherence.
            double R_phase = std::arg(R_eTE);
            double T_phase = std::arg(1.0 + R_eTE);
            R_eTE = std::polar(std::sqrt(refl), R_phase);
            R_eTM = R_eTE;
            T_eTE = std::polar(std::sqrt(1.0 - refl), T_phase);
            T_eTM = T_eTE;
            reflection_gain = refl;
            refraction_gain = 1.0 - refl;
        }

        if (geometry_type == 1 && dense2light && !is_scalar)
            T_eTE = 1.0, T_eTM = 1.0, refraction_gain = 1.0, reflection_gain = 0.0;

        // Select corresponding type
        double eTE_Re = (geometry_type == 0) ? std::real(R_eTE) : std::real(T_eTE),
               eTE_Im = (geometry_type == 0) ? std::imag(R_eTE) : std::imag(T_eTE),
               eTM_Re = (geometry_type == 0) ? std::real(R_eTM) : std::real(T_eTM),
               eTM_Im = (geometry_type == 0) ? std::imag(R_eTM) : std::imag(T_eTM);

        // Read the output ray index
        size_t i_rayN = output_ray_index[iRx] - 1; // Output ray index, 0-based
        if (i_rayN >= n_rayN_t)                    // Just to be sure to avoid any segfaults
            throw std::invalid_argument("Something went wrong. This should never be reached!");

        // Write origN, add a small offset to prevent it from getting stuck inside the mesh element
        if (p_origN != nullptr)
        {
            p_origN[i_rayN] = dtype(Fx + ray_offset * FDx);
            p_origN[i_rayN + n_rayN_t] = dtype(Fy + ray_offset * FDy);
            p_origN[i_rayN + 2 * n_rayN_t] = dtype(Fz + ray_offset * FDz);
        }

        // Write destN
        if (p_destN != nullptr)
        {
            // Make sure the new destination is beyond the new start point
            FD_length = (FD_length <= ray_offset) ? 2.0 * ray_offset : FD_length;
            p_destN[i_rayN] = dtype(Fx + FD_length * FDx);
            p_destN[i_rayN + n_rayN_t] = dtype(Fy + FD_length * FDy);
            p_destN[i_rayN + 2 * n_rayN_t] = dtype(Fz + FD_length * FDz);
        }

        if (p_xprmatN || p_gainN)
        {
            if (is_scalar)
            {
                double amplitude = std::sqrt(gain);
                double coeff_Re = amplitude * eTE_Re;
                double coeff_Im = amplitude * eTE_Im;

                if (p_xprmatN)
                {
                    p_xprmatN[i_rayN] = (dtype)coeff_Re;
                    p_xprmatN[i_rayN + n_rayN_t] = (dtype)coeff_Im;
                    p_xprmatN[i_rayN + 2 * n_rayN_t] = (dtype)0.0;
                    p_xprmatN[i_rayN + 3 * n_rayN_t] = (dtype)0.0;
                    p_xprmatN[i_rayN + 4 * n_rayN_t] = (dtype)0.0;
                    p_xprmatN[i_rayN + 5 * n_rayN_t] = (dtype)0.0;
                    p_xprmatN[i_rayN + 6 * n_rayN_t] = (dtype)0.0;
                    p_xprmatN[i_rayN + 7 * n_rayN_t] = (dtype)0.0;
                }
                if (p_gainN)
                    p_gainN[i_rayN] = (dtype)(coeff_Re * coeff_Re + coeff_Im * coeff_Im);
            }
            else
            {
                // Calculate vectors for the polarization base transformation (incoming path)
                double Hx = -OFy + 3.0e-20, Hy = OFx, Hz = 0.0;                                                // Polarization base in ePhi direction (horizontal)
                scl = 1.0 / std::sqrt(Hx * Hx + Hy * Hy), Hx *= scl, Hy *= scl;                                // Normalize
                double Vx = -OFz * Hy, Vy = OFz * Hx, Vz = OFx * Hy - OFy * Hx;                                // Polarization base in eTheta direction (vertical)
                double Qx = OFy * Nz - OFz * Ny + 3.0e-20, Qy = OFz * Nx - OFx * Nz, Qz = OFx * Ny - OFy * Nx; // Base vector perpendicular to plane normal (eQ)
                scl = 1.0 / std::sqrt(Qx * Qx + Qy * Qy + Qz * Qz), Qx *= scl, Qy *= scl, Qz *= scl;           // Normalize
                double Px = Qy * OFz - Qz * OFy, Py = Qz * OFx - Qx * OFz, Pz = Qx * OFy - Qy * OFx;           // Base vector parallel to plane normal (eP)

                // Calculate polarization base transformation matrix from global coordinates to local coordinates
                bool do_base_transform = scl < 1.0e19;
                double Q1 = (do_base_transform) ? Vx * Px + Vy * Py + Vz * Pz : 1.0; // dot( eV, eP )
                double Q2 = (do_base_transform) ? Vx * Qx + Vy * Qy + Vz * Qz : 0.0; // dot( eV, eQ )
                double Q3 = (do_base_transform) ? Hx * Px + Hy * Py + Hz * Pz : 0.0; // dot( eH, eP )
                double Q4 = (do_base_transform) ? Hx * Qx + Hy * Qy + Hz * Qz : 1.0; // dot( eH, eQ )

                // Calculate vectors for the polarization base transformation (outgoing path)
                Hx = -FDy + 3.0e-20, Hy = FDx, Hz = 0.0;                                                // Polarization base in ePhi direction (horizontal)
                scl = 1.0 / std::sqrt(Hx * Hx + Hy * Hy), Hx *= scl, Hy *= scl;                         // Normalize
                Vx = -FDz * Hy, Vy = FDz * Hx, Vz = FDx * Hy - FDy * Hx;                                // Polarization base in eTheta direction (vertical)
                Qx = FDy * Nz - FDz * Ny + 3.0e-20, Qy = FDz * Nx - FDx * Nz, Qz = FDx * Ny - FDy * Nx; // Base vector perpendicular to plane normal (eQ)
                scl = 1.0 / std::sqrt(Qx * Qx + Qy * Qy + Qz * Qz), Qx *= scl, Qy *= scl, Qz *= scl;    // Normalize
                Px = Qy * FDz - Qz * FDy, Py = Qz * FDx - Qx * FDz, Pz = Qx * FDy - Qy * FDx;           // Base vector parallel to plane normal (eP)

                // Calculate polarization base transformation matrix from global coordinates to local coordinates
                do_base_transform = scl < 1.0e19;
                double U1 = (do_base_transform) ? Vx * Px + Vy * Py + Vz * Pz : 1.0; // dot( eV, eP )
                double U2 = (do_base_transform) ? Vx * Qx + Vy * Qy + Vz * Qz : 0.0; // dot( eV, eQ )
                double U3 = (do_base_transform) ? Hx * Px + Hy * Py + Hz * Pz : 0.0; // dot( eH, eP )
                double U4 = (do_base_transform) ? Hx * Qx + Hy * Qy + Hz * Qz : 1.0; // dot( eH, eQ )

                // Calculate polarization transfer matrix
                // Note: eTE = perpendicular to face normal vector = Horizontal polarization
                //       eTM = parallel to face normal vector = Vertical polarization
                double amplitude = std::sqrt(gain); // Reduction in amplitude caused by conductive medium
                if (interaction_type == 1)          // Scale amplitude in case of transmission
                    amplitude *= std::sqrt((1.0 - reflection_gain) / refraction_gain);

                double VV_Re = amplitude * (U1 * Q1 * eTM_Re + U3 * Q2 * eTE_Re),
                       VV_Im = amplitude * (U1 * Q1 * eTM_Im + U3 * Q2 * eTE_Im),
                       HV_Re = amplitude * (U2 * Q1 * eTM_Re + U4 * Q2 * eTE_Re),
                       HV_Im = amplitude * (U2 * Q1 * eTM_Im + U4 * Q2 * eTE_Im),
                       VH_Re = amplitude * (U1 * Q3 * eTM_Re + U3 * Q4 * eTE_Re),
                       VH_Im = amplitude * (U1 * Q3 * eTM_Im + U3 * Q4 * eTE_Im),
                       HH_Re = amplitude * (U2 * Q3 * eTM_Re + U4 * Q4 * eTE_Re),
                       HH_Im = amplitude * (U2 * Q3 * eTM_Im + U4 * Q4 * eTE_Im);

                // Write XPRMAT
                if (p_xprmatN)
                {
                    p_xprmatN[i_rayN] = (dtype)VV_Re;
                    p_xprmatN[i_rayN + n_rayN_t] = (dtype)VV_Im;
                    p_xprmatN[i_rayN + 2 * n_rayN_t] = (dtype)HV_Re;
                    p_xprmatN[i_rayN + 3 * n_rayN_t] = (dtype)HV_Im;
                    p_xprmatN[i_rayN + 4 * n_rayN_t] = (dtype)VH_Re;
                    p_xprmatN[i_rayN + 5 * n_rayN_t] = (dtype)VH_Im;
                    p_xprmatN[i_rayN + 6 * n_rayN_t] = (dtype)HH_Re;
                    p_xprmatN[i_rayN + 7 * n_rayN_t] = (dtype)HH_Im;
                }
                if (p_gainN)
                    p_gainN[i_rayN] = (dtype)(0.5 * (VV_Re * VV_Re + VV_Im * VV_Im +
                                                     HV_Re * HV_Re + HV_Im * HV_Im +
                                                     VH_Re * VH_Re + VH_Im * VH_Im +
                                                     HH_Re * HH_Re + HH_Im * HH_Im));
            }
        }

        // Write trivecN
        if (use_ray_tube && p_trivecN != nullptr)
        {
            p_trivecN[i_rayN] = (dtype)p_trivec_tmp[0];
            p_trivecN[i_rayN + n_rayN_t] = (dtype)p_trivec_tmp[1];
            p_trivecN[i_rayN + 2 * n_rayN_t] = (dtype)p_trivec_tmp[2];
            p_trivecN[i_rayN + 3 * n_rayN_t] = (dtype)p_trivec_tmp[3];
            p_trivecN[i_rayN + 4 * n_rayN_t] = (dtype)p_trivec_tmp[4];
            p_trivecN[i_rayN + 5 * n_rayN_t] = (dtype)p_trivec_tmp[5];
            p_trivecN[i_rayN + 6 * n_rayN_t] = (dtype)p_trivec_tmp[6];
            p_trivecN[i_rayN + 7 * n_rayN_t] = (dtype)p_trivec_tmp[7];
            p_trivecN[i_rayN + 8 * n_rayN_t] = (dtype)p_trivec_tmp[8];
        }

        // Write tridirN
        if (use_ray_tube == 1 && p_tridirN != nullptr)
        {
            p_tridirN[i_rayN] = (dtype)p_tridir_tmp[0];
            p_tridirN[i_rayN + n_rayN_t] = (dtype)p_tridir_tmp[1];
            p_tridirN[i_rayN + 2 * n_rayN_t] = (dtype)p_tridir_tmp[2];
            p_tridirN[i_rayN + 3 * n_rayN_t] = (dtype)p_tridir_tmp[3];
            p_tridirN[i_rayN + 4 * n_rayN_t] = (dtype)p_tridir_tmp[4];
            p_tridirN[i_rayN + 5 * n_rayN_t] = (dtype)p_tridir_tmp[5];
        }
        else if (use_ray_tube == 2 && p_tridirN != nullptr)
        {
            p_tridirN[i_rayN] = (dtype)p_tridir_tmp[0];
            p_tridirN[i_rayN + n_rayN_t] = (dtype)p_tridir_tmp[1];
            p_tridirN[i_rayN + 2 * n_rayN_t] = (dtype)p_tridir_tmp[2];
            p_tridirN[i_rayN + 3 * n_rayN_t] = (dtype)p_tridir_tmp[3];
            p_tridirN[i_rayN + 4 * n_rayN_t] = (dtype)p_tridir_tmp[4];
            p_tridirN[i_rayN + 5 * n_rayN_t] = (dtype)p_tridir_tmp[5];
            p_tridirN[i_rayN + 6 * n_rayN_t] = (dtype)p_tridir_tmp[6];
            p_tridirN[i_rayN + 7 * n_rayN_t] = (dtype)p_tridir_tmp[7];
            p_tridirN[i_rayN + 8 * n_rayN_t] = (dtype)p_tridir_tmp[8];
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
            p_normal_vecN[i_rayN + n_rayN_t] = (dtype)Ny;
            p_normal_vecN[i_rayN + 2 * n_rayN_t] = (dtype)Nz;
            p_normal_vecN[i_rayN + 3 * n_rayN_t] = (dtype)Mx;
            p_normal_vecN[i_rayN + 4 * n_rayN_t] = (dtype)My;
            p_normal_vecN[i_rayN + 5 * n_rayN_t] = (dtype)Mz;
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

// ray_state_update — support code.
//
// The helpers below back the inside/outside ray-state machine ported out of
// calc_diffraction_gain.cpp (its dispatch, source lines 495-907) and the analytic thin-slab
// (Fabry-Perot) overlay. They sit in the same translation unit as ray_mesh_interact and the
// merged material helpers above, so no extra include is required.
//
// State / material-index convention (see Section 7 of the design spec):
//   - Material indices (M1 = mtl_ind_fbs, M2 = mtl_ind_sbs, and the three state words) index
//     mtl_prop directly. The value 0 is reserved for "outside / air / empty"; real materials use
//     indices >= 1. medium_gain_impl / transition_gain_linear therefore receive the index as-is
//     and a value of 0 maps to air (gain 1, unit index). This reserves mtl_prop[0] for air and is
//     what makes the port bit-identical to calc_diffraction_gain, whose state stores 1-based face
//     indices that resolve to the same material indices.
//   - State words are bit-masked, never abs(): mat = w & 0x7FFF, flag = w & 0x8000. A flag is set
//     by OR, (short)(X | 0x8000), never by negation.

// Check whether two media are the same material. Material-index variant of the former
// face-index helper (the mtl_ind argument is dropped, Section 11): on indices this reduces to a
// plain equality, which is exactly the face-table comparison the source performed.
static inline bool same_materials(int iMa, int iMb) // material indices (0 = air/none)
{
    return iMa == iMb;
}

// Medium-to-medium transition gain (transmission power gain across one interface, including the
// lumped interface gain). Material-index variant of the former face-index helper (the mtl_ind
// argument is dropped, Section 11): iMa / iMb are material indices, 0 = air. The body is the port
// of transition_gain_linear from calc_diffraction_gain.cpp with face->material resolution removed.
template <typename dtype>
static inline dtype transition_gain_linear(const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
                                           int iMa, int iMb, // material indices (0 = air/none)
                                           dtype theta, dtype fGHz, bool is_scalar)
{
    if (mtl_prop == nullptr) // No material model: air-to-air, full transmission
        return (dtype)1.0;

    // Resolve named material columns (nullptr -> default applied)
    const dtype *m_a = mtl_col(mtl_prop, "a");
    const dtype *m_b = mtl_col(mtl_prop, "b");
    const dtype *m_c = mtl_col(mtl_prop, "c");
    const dtype *m_d = mtl_col(mtl_prop, "d");
    const dtype *m_e = mtl_col(mtl_prop, "e");
    const dtype *m_f = mtl_col(mtl_prop, "f");
    const dtype *m_g = mtl_col(mtl_prop, "g");
    const dtype *m_h = mtl_col(mtl_prop, "h");
    const dtype *m_fRef = mtl_col(mtl_prop, "fRef");
    const dtype *m_resF = mtl_col(mtl_prop, "resF");
    const dtype *m_resQ = mtl_col(mtl_prop, "resQ");
    const dtype *m_resS = mtl_col(mtl_prop, "resS");
    const dtype *m_tf = mtl_col(mtl_prop, "tf");
    const dtype *m_tfB = mtl_col(mtl_prop, "tfB");

    // Material indices used directly (0 = air, handled by the iMa/iMb != 0 guards below)
    arma::uword mF = (arma::uword)((iMa < 0) ? 0 : iMa);
    arma::uword mS = (arma::uword)((iMb < 0) ? 0 : iMb);

    // Convert to double
    double dTheta = (double)theta;

    // Limit value to 0 ... 1 for calculating reflection and transmission coefficients
    double abs_cos_theta = std::abs(std::cos(dTheta + 1.570796326794897));
    abs_cos_theta = (abs_cos_theta > 1.0) ? 1.0 : abs_cos_theta;
    double sin_theta = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta); // Trigonometric identity

    // Defaults: air for both media
    double kR1 = 1.0, kR2 = 0.0, kR3 = 0.0, kR4 = 0.0, kR_fRef = 1.0;
    double kR5 = 1.0, kR6 = 0.0, kR7 = 0.0, kR8 = 0.0;
    double kS1 = 1.0, kS2 = 0.0, kS3 = 0.0, kS4 = 0.0, kS_fRef = 1.0;
    double kS5 = 1.0, kS6 = 0.0, kS7 = 0.0, kS8 = 0.0;
    double kR_resF = 0.0, kR_resQ = 0.0, kR_resS = 0.0;
    double kS_resF = 0.0, kS_resQ = 0.0, kS_resS = 0.0;
    double kR_tf = 0.0, kR_tfB = 0.0, kS_tf = 0.0, kS_tfB = 0.0;
    double transition_gain = 1.0;

    if (iMa != 0)
    {
        if (dTheta >= 0.0) // Ray hits front side of FBS/SBS face, set second material to object material
        {
            kS1 = mtl_val(m_a, mF, 1.0);
            kS2 = mtl_val(m_b, mF, 0.0);
            kS3 = mtl_val(m_c, mF, 0.0);
            kS4 = mtl_val(m_d, mF, 0.0);
            kS5 = mtl_val(m_e, mF, 1.0);
            kS6 = mtl_val(m_f, mF, 0.0);
            kS7 = mtl_val(m_g, mF, 0.0);
            kS8 = mtl_val(m_h, mF, 0.0);
            kS_fRef = mtl_val(m_fRef, mF, 1.0);
            kS_resF = mtl_val(m_resF, mF, 0.0);
            kS_resQ = mtl_val(m_resQ, mF, 0.0);
            kS_resS = mtl_val(m_resS, mF, 0.0);
            kS_tf = mtl_val(m_tf, mF, 0.0);
            kS_tfB = mtl_val(m_tfB, mF, 0.0);
            transition_gain = (double)interface_gain_impl(mtl_prop, mF, fGHz * (dtype)1e9);
        }
        else // Ray hits back side of FBS face, set first material to object material
        {
            kR1 = mtl_val(m_a, mF, 1.0);
            kR2 = mtl_val(m_b, mF, 0.0);
            kR3 = mtl_val(m_c, mF, 0.0);
            kR4 = mtl_val(m_d, mF, 0.0);
            kR5 = mtl_val(m_e, mF, 1.0);
            kR6 = mtl_val(m_f, mF, 0.0);
            kR7 = mtl_val(m_g, mF, 0.0);
            kR8 = mtl_val(m_h, mF, 0.0);
            kR_fRef = mtl_val(m_fRef, mF, 1.0);
            kR_resF = mtl_val(m_resF, mF, 0.0);
            kR_resQ = mtl_val(m_resQ, mF, 0.0);
            kR_resS = mtl_val(m_resS, mF, 0.0);
            kR_tf = mtl_val(m_tf, mF, 0.0);
            kR_tfB = mtl_val(m_tfB, mF, 0.0);
        }
    }

    if (iMb != 0) // Material to material transition
    {
        if (dTheta >= 0.0) // SBS (front side) is hit first
        {
            kR1 = mtl_val(m_a, mS, 1.0);
            kR2 = mtl_val(m_b, mS, 0.0);
            kR3 = mtl_val(m_c, mS, 0.0);
            kR4 = mtl_val(m_d, mS, 0.0);
            kR5 = mtl_val(m_e, mS, 1.0);
            kR6 = mtl_val(m_f, mS, 0.0);
            kR7 = mtl_val(m_g, mS, 0.0);
            kR8 = mtl_val(m_h, mS, 0.0);
            kR_fRef = mtl_val(m_fRef, mS, 1.0);
            kR_resF = mtl_val(m_resF, mS, 0.0);
            kR_resQ = mtl_val(m_resQ, mS, 0.0);
            kR_resS = mtl_val(m_resS, mS, 0.0);
            kR_tf = mtl_val(m_tf, mS, 0.0);
            kR_tfB = mtl_val(m_tfB, mS, 0.0);
        }
        else // FBS (back side) is hit first
        {
            kS1 = mtl_val(m_a, mS, 1.0);
            kS2 = mtl_val(m_b, mS, 0.0);
            kS3 = mtl_val(m_c, mS, 0.0);
            kS4 = mtl_val(m_d, mS, 0.0);
            kS5 = mtl_val(m_e, mS, 1.0);
            kS6 = mtl_val(m_f, mS, 0.0);
            kS7 = mtl_val(m_g, mS, 0.0);
            kS8 = mtl_val(m_h, mS, 0.0);
            kS_fRef = mtl_val(m_fRef, mS, 1.0);
            kS_resF = mtl_val(m_resF, mS, 0.0);
            kS_resQ = mtl_val(m_resQ, mS, 0.0);
            kS_resS = mtl_val(m_resS, mS, 0.0);
            kS_tf = mtl_val(m_tf, mS, 0.0);
            kS_tfB = mtl_val(m_tfB, mS, 0.0);
            transition_gain = (double)interface_gain_impl(mtl_prop, mS, fGHz * (dtype)1e9);
        }
    }

    // Calculate complex-valued relative permittivity of medium 1 and 2, ITU-R P.2040-1, eq. (9b)
    std::complex<double> eta1 = eta_from_coeffs(kR1, kR2, kR3, kR4, kR_fRef, (double)fGHz) +
                                eta_resonance(kR_resF, kR_resQ, kR_resS, (double)fGHz);

    std::complex<double> eta2 = eta_from_coeffs(kS1, kS2, kS3, kS4, kS_fRef, (double)fGHz) +
                                eta_resonance(kS_resF, kS_resQ, kS_resS, (double)fGHz);

    std::complex<double> mu1 = mu_from_coeffs(kR5, kR6, kR7, kR8, kR_fRef, (double)fGHz);
    std::complex<double> mu2 = mu_from_coeffs(kS5, kS6, kS7, kS8, kS_fRef, (double)fGHz);

    bool dense2light = std::real(eta1 * mu1) > std::real(eta2 * mu2);

    double reflection_gain = 0.0;
    if (is_scalar)
    {
        // Scalar: physical Fresnel reflection (TE), redistributed by the transmission factor.
        // Energy conserved by construction (refl + trans = 1); no dense2light pass-through.
        std::complex<double> eta1_div_eta2 = (eta1 * mu1) / (eta2 * mu2);
        std::complex<double> cos_theta2 = std::sqrt(1.0 - eta1_div_eta2 * sin_theta * sin_theta);
        std::complex<double> z1 = std::sqrt(eta1 / mu1);
        std::complex<double> z2 = std::sqrt(eta2 / mu2);
        std::complex<double> R_eTE = (z1 * abs_cos_theta - z2 * cos_theta2) /
                                     (z1 * abs_cos_theta + z2 * cos_theta2);
        double tf_eff = tf_value((dTheta >= 0.0) ? kS_tf : kR_tf,
                                 (dTheta >= 0.0) ? kS_tfB : kR_tfB,
                                 (dTheta >= 0.0) ? kS_fRef : kR_fRef, (double)fGHz);
        reflection_gain = tf_apply(std::norm(R_eTE), tf_eff);
    }
    else if (!dense2light) // EM: Fresnel on light->dense, pass-through on dense->light
    {
        std::complex<double> eta1_div_eta2 = (eta1 * mu1) / (eta2 * mu2);
        std::complex<double> cos_theta2 = std::sqrt(1.0 - eta1_div_eta2 * sin_theta * sin_theta);
        eta1 = std::sqrt(eta1 / mu1);
        eta2 = std::sqrt(eta2 / mu2);
        std::complex<double> R_eTE = (eta1 * abs_cos_theta - eta2 * cos_theta2) /
                                     (eta1 * abs_cos_theta + eta2 * cos_theta2);
        std::complex<double> R_eTM = (eta2 * abs_cos_theta - eta1 * cos_theta2) /
                                     (eta2 * abs_cos_theta + eta1 * cos_theta2);
        reflection_gain = 0.5 * (std::norm(R_eTE) + std::norm(R_eTM));
    }
    return dtype(transition_gain * (1.0 - reflection_gain));
}

// Parallelism gate (Section 9.3). Two faces bound a slab when their planes share orientation,
// regardless of normal sign: the magnitude of the normal dot product is tested, so both
// dot ~ +1 and dot ~ -1 count as parallel. Only genuine wedges/edges (|dot| well below 1) fail.
// tol = 3.8e-3 (~5 deg).
static inline bool faces_parallel(double nfx, double nfy, double nfz,
                                  double nsx, double nsy, double nsz)
{
    const double tol = 3.8e-3;
    double d = nfx * nsx + nfy * nsy + nfz * nsz;
    return std::abs(d) > 1.0 - tol;
}

// Analytic thin-slab (Fabry-Perot) factor S = 1 / (1 - r_near * r_far * phi^2) with the survival
// gate (Section 9.4), the near-pole clamp (Section 9.7) and tf-effective, Stokes-consistent
// coefficients (Section 9.5). Returns true and writes S when the slab is resolved; returns false
// (re-emit) when the parallelism flag is set, when the round-trip amplitude rho is below eps, or
// when the denominator is near the pole. On a false return S_re / S_im are left untouched.
//
//   slab_mat   medium inside the cavity (Section 9.1: the current medium; air, index 0, for the
//              i-o-i air-gap case). For air, |phi| = 1 and the index is unity.
//   near_mat   medium on the far side of the interface being processed (r_near, slab side).
//   far_mat    medium on the far side of the opposite interface (r_far).
//   theta      incidence angle in the fbs_angleN convention (theta = acos(cos_inc) - pi/2).
//   L          one-way in-slab path d(orig, fbs).
//
// TODO(QRT): the exact r_near/r_far adjacent-material assignment and the in-slab one-way
// loss/phase ownership (dispatch MED vs QRT geometric per-segment, Section 9.2) are finalized at
// tracer integration. S supplies only the round-trip resonant factor phi^2 inside the series.
template <typename dtype>
static inline bool slab_airy_factor(const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
                                    int slab_mat, int near_mat, int far_mat,
                                    double theta, double L, dtype center_frequency,
                                    bool is_scalar, bool parallel_ok, double eps,
                                    double &S_re, double &S_im)
{
    (void)is_scalar;  // the TE Fresnel form below already matches the scalar interface model
    if (!parallel_ok) // known wedge/edge -> re-emit
        return false;

    const double fGHz = (double)center_frequency * 1e-9;
    const double c0 = 299792458.0;
    const double omega = 2.0 * 3.14159265358979323846 * (double)center_frequency;

    // Resolve named material columns once (nullptr -> per-column air defaults)
    const dtype *m_a = mtl_prop ? mtl_col(mtl_prop, "a") : nullptr;
    const dtype *m_b = mtl_prop ? mtl_col(mtl_prop, "b") : nullptr;
    const dtype *m_c = mtl_prop ? mtl_col(mtl_prop, "c") : nullptr;
    const dtype *m_d = mtl_prop ? mtl_col(mtl_prop, "d") : nullptr;
    const dtype *m_e = mtl_prop ? mtl_col(mtl_prop, "e") : nullptr;
    const dtype *m_f = mtl_prop ? mtl_col(mtl_prop, "f") : nullptr;
    const dtype *m_g = mtl_prop ? mtl_col(mtl_prop, "g") : nullptr;
    const dtype *m_h = mtl_prop ? mtl_col(mtl_prop, "h") : nullptr;
    const dtype *m_fRef = mtl_prop ? mtl_col(mtl_prop, "fRef") : nullptr;
    const dtype *m_resF = mtl_prop ? mtl_col(mtl_prop, "resF") : nullptr;
    const dtype *m_resQ = mtl_prop ? mtl_col(mtl_prop, "resQ") : nullptr;
    const dtype *m_resS = mtl_prop ? mtl_col(mtl_prop, "resS") : nullptr;
    const dtype *m_tf = mtl_prop ? mtl_col(mtl_prop, "tf") : nullptr;
    const dtype *m_tfB = mtl_prop ? mtl_col(mtl_prop, "tfB") : nullptr;

    // eta_if: resonance-included permittivity (interface / Fresnel). eta_med: resonance-excluded
    // permittivity (medium path / phase). mu: relative permeability. tfv: transmission factor.
    auto resolve_eta = [&](int mat, std::complex<double> &eta_if, std::complex<double> &mu,
                           std::complex<double> &eta_med, double &tfv)
    {
        if (mat == 0 || mtl_prop == nullptr) // air / vacuum
        {
            eta_if = std::complex<double>(1.0, 0.0);
            mu = std::complex<double>(1.0, 0.0);
            eta_med = std::complex<double>(1.0, 0.0);
            tfv = 0.0;
            return;
        }
        arma::uword im = (arma::uword)mat;
        std::complex<double> e0 = eta_from_coeffs(mtl_val(m_a, im, 1.0), mtl_val(m_b, im, 0.0),
                                                  mtl_val(m_c, im, 0.0), mtl_val(m_d, im, 0.0),
                                                  mtl_val(m_fRef, im, 1.0), fGHz);
        mu = mu_from_coeffs(mtl_val(m_e, im, 1.0), mtl_val(m_f, im, 0.0),
                            mtl_val(m_g, im, 0.0), mtl_val(m_h, im, 0.0),
                            mtl_val(m_fRef, im, 1.0), fGHz);
        eta_med = e0;
        eta_if = e0 + eta_resonance(mtl_val(m_resF, im, 0.0), mtl_val(m_resQ, im, 0.0),
                                    mtl_val(m_resS, im, 0.0), fGHz);
        tfv = tf_value(mtl_val(m_tf, im, 0.0), mtl_val(m_tfB, im, 0.0), mtl_val(m_fRef, im, 1.0), fGHz);
    };

    // Slab medium
    std::complex<double> eta_s_if, mu_s, eta_s_med;
    double tf_s;
    resolve_eta(slab_mat, eta_s_if, mu_s, eta_s_med, tf_s);

    // Incidence cosine (fbs_angleN convention)
    double abs_cos = std::abs(std::cos(theta + 1.570796326794897));
    abs_cos = (abs_cos > 1.0) ? 1.0 : abs_cos;
    double sin2 = 1.0 - abs_cos * abs_cos;

    // Fresnel (TE) amplitude reflection at slab|adjacent from the slab side, with tf folded into
    // the magnitude and the Fresnel phase preserved (Section 9.5). Returns r and R = |r|^2.
    auto fresnel_r = [&](int adj_mat, std::complex<double> &r, double &R)
    {
        std::complex<double> eta_a_if, mu_a, eta_a_med;
        double tf_a;
        resolve_eta(adj_mat, eta_a_if, mu_a, eta_a_med, tf_a);
        std::complex<double> z1 = std::sqrt(eta_s_if / mu_s); // slab admittance
        std::complex<double> z2 = std::sqrt(eta_a_if / mu_a); // adjacent admittance
        std::complex<double> ratio = (eta_s_if * mu_s) / (eta_a_if * mu_a);
        std::complex<double> cos_t2 = std::sqrt(1.0 - ratio * sin2);
        std::complex<double> r_te = (z1 * abs_cos - z2 * cos_t2) / (z1 * abs_cos + z2 * cos_t2);
        double R0 = std::norm(r_te);
        double Reff = tf_apply(R0, tf_a); // tf carried by the adjacent (entered) material
        r = std::polar(std::sqrt(Reff), std::arg(r_te));
        R = Reff;
    };

    std::complex<double> r_near, r_far;
    double R_near = 0.0, R_far = 0.0;
    fresnel_r(near_mat, r_near, R_near);
    fresnel_r(far_mat, r_far, R_far);

    // One-way in-slab propagation phi (Section 9.1): magnitude from the full medium_gain (dielectric
    // + alpha + mass), phase from the resonance-excluded permittivity only. Air slab -> lossless,
    // unit index.
    double gL, n_re;
    if (slab_mat == 0 || mtl_prop == nullptr)
        gL = 1.0, n_re = 1.0;
    else
    {
        gL = (double)medium_gain_impl(mtl_prop, (arma::uword)slab_mat, (dtype)L, center_frequency);
        n_re = std::real(std::sqrt(eta_s_med * mu_s)); // real refractive index
    }
    double abs_phi = std::sqrt((gL < 0.0) ? 0.0 : gL);
    double arg_phi = -(omega / c0) * n_re * L;
    std::complex<double> phi2 = std::polar(abs_phi * abs_phi, 2.0 * arg_phi); // phi^2

    std::complex<double> denom = std::complex<double>(1.0, 0.0) - r_near * r_far * phi2;

    // Survival gate (Section 9.4): rho^2 = R_near * R_far * medium_gain(2L)
    double g2L = (slab_mat == 0 || mtl_prop == nullptr)
                     ? 1.0
                     : (double)medium_gain_impl(mtl_prop, (arma::uword)slab_mat, (dtype)(2.0 * L), center_frequency);
    double rr = R_near * R_far;
    rr = (rr < 0.0) ? 0.0 : rr;
    g2L = (g2L < 0.0) ? 0.0 : g2L;
    double rho = std::sqrt(rr * g2L);

    // Survival + near-pole clamp (Section 9.7): hand the near-pole / low-amplitude case back to
    // the tracer as a re-emit.
    if (rho < eps || std::abs(denom) < 1.0e-2)
        return false;

    std::complex<double> S = std::complex<double>(1.0, 0.0) / denom;
    S_re = std::real(S);
    S_im = std::imag(S);
    return true;
}

// Gain / xprmat patch operations (Section 5). xprmatN columns are VV(0,1) HV(2,3) VH(4,5) HH(6,7),
// re/im per entry. Either output may be null; the other is still patched.

// Uniform complex scale by c = (cr + j*ci): multiply each Jones entry by c and keep gainN
// consistent (gainN *= |c|^2). Backs IG (c = 1), IG*MED (c = sqrt(g)), IG*S (c = S) and
// IG*S*MED (c = S*sqrt(g)). Mode-agnostic: scalar mode keeps its zero off-diagonal entries zero.
// With xprmatN == nullptr an IG*S row degrades to the magnitude-only gainN *= |S|^2 (Section 5).
template <typename dtype>
static inline void rsu_scale(dtype *p_xprmatN, dtype *p_gainN, size_t i, size_t n_rayN_t,
                             double cr, double ci)
{
    if (p_xprmatN != nullptr)
        for (int k = 0; k < 4; ++k)
        {
            size_t re_i = i + (size_t)(2 * k) * n_rayN_t;
            size_t im_i = i + (size_t)(2 * k + 1) * n_rayN_t;
            double re = (double)p_xprmatN[re_i];
            double im = (double)p_xprmatN[im_i];
            p_xprmatN[re_i] = (dtype)(re * cr - im * ci);
            p_xprmatN[im_i] = (dtype)(re * ci + im * cr);
        }
    if (p_gainN != nullptr)
        p_gainN[i] = (dtype)((double)p_gainN[i] * (cr * cr + ci * ci));
}

// Isotropic replace with in-medium / transition power gain g (field sqrt(g)): discards the
// (spurious) interaction. EM lays sqrt(g) on VV and HH (gainN = 0.5*(g+g) = g); scalar lays it on
// VV only (gainN = |VV|^2 = g). Backs the MED(...) and MED*TRN*MED rows. gainN is set to g to stay
// bit-identical with the diffraction reference even where the unclamped path yields g > 1.
template <typename dtype>
static inline void rsu_replace(dtype *p_xprmatN, dtype *p_gainN, size_t i, size_t n_rayN_t,
                               double g, bool is_scalar)
{
    if (p_xprmatN != nullptr)
    {
        for (int c = 0; c < 8; ++c)
            p_xprmatN[i + (size_t)c * n_rayN_t] = (dtype)0.0;
        double a = std::sqrt((g < 0.0) ? 0.0 : g);
        p_xprmatN[i] = (dtype)a; // VV_re
        if (!is_scalar)
            p_xprmatN[i + (size_t)6 * n_rayN_t] = (dtype)a; // HH_re
    }
    if (p_gainN != nullptr)
        p_gainN[i] = (dtype)g;
}

// KILL: zero the interaction (Section 5).
template <typename dtype>
static inline void rsu_kill(dtype *p_xprmatN, dtype *p_gainN, size_t i, size_t n_rayN_t)
{
    if (p_xprmatN != nullptr)
        for (int c = 0; c < 8; ++c)
            p_xprmatN[i + (size_t)c * n_rayN_t] = (dtype)0.0;
    if (p_gainN != nullptr)
        p_gainN[i] = (dtype)0.0;
}

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
    arma::Col<short> *mtl_ind_prev_out = nullptr,
    arma::Col<short> *mtl_ind_current_out = nullptr,
    arma::Col<short> *mtl_ind_buffer_out = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::u32_vec *ray_ind = nullptr,
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

    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");

    if (orig == nullptr || dest == nullptr || fbs == nullptr || sbs == nullptr)
        throw std::invalid_argument("Inputs 'orig', 'dest', 'fbs' and 'sbs' cannot be NULL.");
    if (out_typeN == nullptr)
        throw std::invalid_argument("Input 'out_typeN' cannot be NULL.");

    // Validate the material map once; internal mtl_col / mtl_val accesses are then safe for any
    // material index < n_mtl (the caller guarantees M1, M2 and the state words are in range).
    if (mtl_prop != nullptr)
        (void)mtl_validate(*mtl_prop);

    const bool is_scalar = interaction_type >= 3;
    const bool refl_pass = (interaction_type == 0 || interaction_type == 3); // geometry 0

    const arma::uword n_rayN = out_typeN->n_elem;
    if (n_rayN >= INT32_MAX)
        throw std::invalid_argument("Number of interaction rays exceeds maximum supported number.");
    const size_t n_rayN_t = (size_t)n_rayN;
    const int n_rayN_i = (int)n_rayN;

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
    for (int i = 0; i < n_rayN_i; ++i) // Interaction loop (compact set)
    {
        size_t ii = (size_t)i;
        size_t g = (p_ray_ind == nullptr) ? ii : (size_t)p_ray_ind[ii]; // Full-set index

        // Old state at g (full set). Defaults to copy-through into the compact outputs at i.
        short s_prev = (p_prev_in == nullptr) ? (short)0 : p_prev_in[g];
        short s_cur = (p_cur_in == nullptr) ? (short)0 : p_cur_in[g];
        short s_buf = (p_buf_in == nullptr) ? (short)0 : p_buf_in[g];
        int cur = s_cur & 0x7FFF;
        bool resolved = (s_cur & 0x8000) != 0;
        int buf = s_buf & 0x7FFF;
        int prev_mat = s_prev & 0x7FFF;
        bool prev_nonpar = (s_prev & 0x8000) != 0;
        short out_prev = s_prev, out_cur = s_cur, out_buf = s_buf;

        // Compact-set reads at i
        unsigned nH = (p_no_interact == nullptr) ? 1u : p_no_interact[g];
        int typeH = p_out_typeN[ii];
        int M1 = (p_M1 == nullptr) ? 0 : (int)(p_M1[ii] & (short)0x7FFF);
        int M2 = (p_M2 == nullptr) ? 0 : (int)(p_M2[ii] & (short)0x7FFF);
        double theta = (p_fbs_angleN == nullptr) ? 0.0 : (double)p_fbs_angleN[ii];
        dtype fGHz = (dtype)((double)center_frequency * 1e-9);

        // Euclidean distance between two full-set geometry rows at g
        auto D = [&](const arma::Mat<dtype> *A, const arma::Mat<dtype> *B) -> double
        {
            return (double)qd_calc_length(A->at(g, 0), A->at(g, 1), A->at(g, 2),
                                          B->at(g, 0), B->at(g, 1), B->at(g, 2));
        };

        // Wedge test (Section 9.3): true when FBS and SBS faces sit at a real angle. No-op (false)
        // when normals are absent or the two faces are a single point. Run only at o-i entries that
        // capture both faces (nH >= 2 types 1/7/13).
        auto wedge_nonparallel = [&]() -> bool
        {
            if (p_normal_vecN == nullptr)
                return false;
            if (!(D(fbs, sbs) > 1.0e-6))
                return false;
            double nfx = (double)p_normal_vecN[ii];
            double nfy = (double)p_normal_vecN[ii + n_rayN_t];
            double nfz = (double)p_normal_vecN[ii + 2 * n_rayN_t];
            double nsx = (double)p_normal_vecN[ii + 3 * n_rayN_t];
            double nsy = (double)p_normal_vecN[ii + 4 * n_rayN_t];
            double nsz = (double)p_normal_vecN[ii + 5 * n_rayN_t];
            return !faces_parallel(nfx, nfy, nfz, nsx, nsy, nsz);
        };

        // Medium gain shorthand: MED(m, d) with material index m used directly (0 = air -> 1).
        auto MED = [&](int m, double d) -> double
        {
            return (double)medium_gain_impl(mtl_prop, (arma::uword)((m < 0) ? 0 : m), (dtype)d, center_frequency);
        };
        auto TRN = [&](int a, int b) -> double
        {
            return (double)transition_gain_linear(mtl_prop, a, b, (dtype)theta, fGHz, is_scalar);
        };

        if (refl_pass) // Reflection pass, interaction_type in {0, 3} (Section 10.0, 10.7)
        {
            if (resolved)
            {
                // Resolved-ray reflection: the front reflection is already summed inside S -> KILL.
                rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
            }
            else if (cur == 0)
            {
                // Entry / order-0 front reflection: bare Fresnel r12 (naturally |R| = 1 under TIR).
                // IG, state copy-through.
            }
            else
            {
                // Internal / back reflection of a resolvable parallel slab: r23 * S, set the
                // RESOLVED flag so the ray then exits the front transparently (Section 9.2, 10.7).
                double L = D(orig, fbs);
                double Sre = 0.0, Sim = 0.0;
                bool res = slab_airy_factor(mtl_prop, cur, 0, 0, theta, L, center_frequency,
                                            is_scalar, !prev_nonpar, eps, Sre, Sim);
                if (res)
                {
                    rsu_scale(p_xprmatN, p_gainN, ii, n_rayN_t, Sre, Sim); // IG * S
                    out_cur = (short)((cur & 0x7FFF) | (int)0x8000);       // set resolved flag
                }
                // else: ordinary reflection / re-emit -> IG, copy-through
            }
        }
        else // Transmission / refraction pass, interaction_type in {1, 2, 4}
        {
            bool is_TR = (typeH == 3 || typeH == 6 || typeH == 9 || typeH == 12 || typeH == 15);

            if (is_TR)
            {
                // TR forward-kill (Section 10.6). TR out-codes occur only for interaction_type == 2
                // and win over the resolved flag (Section 10.0): no transmitted field. State unchanged.
                rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
            }
            else if (resolved)
            {
                // Resolved-ray out-coupling (Section 10.0). iM = the next medium for an i-i.
                int iM = (typeH == 5) ? M2 : M1;
                if (typeH == 2 || typeH == 8 || typeH == 14) // i-o: out-coupling t21
                {
                    out_cur = (short)0; // current_out <- 0, clear resolved flag
                }
                else if (typeH == 4 || typeH == 5) // i-i: stay resolved, advance medium
                {
                    out_cur = (short)((iM & 0x7FFF) | (int)0x8000); // keep resolved flag
                    out_prev = (short)cur;                          // prev_out <- old cur
                }
                // else: o-i / edges -> transparent pass-through, IG, state copy-through
            }
            else if (nH == 0)
            {
                // Not processed: the caller applies any whole-segment in-medium loss. Copy-through.
            }
            else if ((nH == 1 && typeH == 1) || (nH == 2 && typeH == 7) || (nH == 2 && typeH == 13))
            {
                // o-i family, entry / overlapping-entry (Section 10.1, Branch A)
                if (cur == 0) // enter
                {
                    double dist = D(fbs, dest);
                    dist = (dist > ray_offset) ? dist - ray_offset : dist;                      // clamped
                    rsu_scale(p_xprmatN, p_gainN, ii, n_rayN_t, std::sqrt(MED(M1, dist)), 0.0); // IG * MED
                    out_cur = (short)M1;
                    bool nonpar = (nH >= 2) && wedge_nonparallel();
                    out_prev = (short)(nonpar ? (int)0x8000 : 0); // prev <- 0, +flag
                }
                else // nested
                {
                    rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, dest)), is_scalar);
                    out_buf = (short)M1;
                }
            }
            else if ((nH == 1 && typeH == 2) || (nH == 2 && typeH == 8) || (nH == 2 && typeH == 14))
            {
                // i-o family, exit / false-inside / virtual transitions (Section 10.2, Branch A)
                if (cur == 0) // false inside
                {
                    // IG, copy-through
                }
                else if (buf == 0) // cavity exit, IG * S
                {
                    double Sre = 0.0, Sim = 0.0;
                    bool res = slab_airy_factor(mtl_prop, cur, 0, 0, theta, D(orig, fbs), center_frequency,
                                                is_scalar, !prev_nonpar, eps, Sre, Sim);
                    if (res)
                        rsu_scale(p_xprmatN, p_gainN, ii, n_rayN_t, Sre, Sim);
                    out_cur = (short)0;
                }
                else if (nH == 1 && typeH == 2) // virtual i-i
                {
                    if (same_materials(buf, M1)) // M2 embedded in M1, ignore M2
                    {
                        rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, dest)), is_scalar);
                        out_buf = (short)0;
                    }
                    else
                    {
                        double g = MED(cur, D(orig, fbs)) * TRN(cur, buf) * MED(buf, D(fbs, dest));
                        rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, g, is_scalar);
                        out_cur = (short)buf;
                        out_buf = (short)0;
                    }
                }
                else // nH == 2 types 8/14, buf != 0: ii-oo
                {
                    double g = MED(cur, D(orig, fbs)) * TRN(cur, 0);
                    rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, g, is_scalar);
                    out_cur = (short)0;
                    out_buf = (short)0;
                }
            }
            else if (nH == 2 && typeH == 1)
            {
                // o-i-o (Section 10.1, Branch B)
                if (cur == 0)
                {
                    // IG (bare); current_out <- M1, +flag
                    out_cur = (short)M1;
                    bool nonpar = wedge_nonparallel();
                    out_prev = (short)(nonpar ? (int)0x8000 : 0);
                }
                else // nested o-i-o
                {
                    rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, fbs)), is_scalar);
                    out_buf = (short)M1;
                }
            }
            else if (nH == 2 && typeH == 2)
            {
                // i-o-i (Section 10.2, Branch B)
                if (buf == 0)
                {
                    if (M2 == 0) // illegal
                        rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
                    else // cavity exit, air gap: slab is air, bounded by M1 / M2 (Section 9.1)
                    {
                        double Sre = 0.0, Sim = 0.0;
                        bool res = slab_airy_factor(mtl_prop, 0, M1, M2, theta, D(orig, fbs), center_frequency,
                                                    is_scalar, !prev_nonpar, eps, Sre, Sim);
                        if (res)
                            rsu_scale(p_xprmatN, p_gainN, ii, n_rayN_t, Sre, Sim);
                        out_cur = (short)0; // survives
                    }
                }
                else if (cur != 0)
                {
                    if (same_materials(buf, M1))
                    {
                        rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, fbs) + ray_offset), is_scalar);
                        out_buf = (short)0; // survives
                    }
                    else
                    {
                        double g = MED(cur, D(orig, fbs)) * TRN(cur, buf) * MED(buf, ray_offset);
                        rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, g, is_scalar);
                        out_cur = (short)buf;
                        out_buf = (short)0; // survives
                    }
                }
                else // buf != 0 and cur == 0: terminate (source lines 649-650)
                    rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
            }
            else if (nH == 2 && (typeH == 4 || typeH == 5))
            {
                // M2M (i-i) family (Section 10.3, nH == 2)
                if (cur == 0) // illegal
                    rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
                else if (buf == 0)
                {
                    if (M1 == 0 || M2 == 0) // illegal
                        rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
                    else // cavity transition: IG * S * MED(iM, d(fbs,dest) - off (unclamped))
                    {
                        int iM = (typeH == 5) ? M2 : M1;
                        double gmed = MED(iM, D(fbs, dest) - ray_offset); // unclamped
                        double Sre = 0.0, Sim = 0.0;
                        bool res = slab_airy_factor(mtl_prop, cur, iM, prev_mat, theta, D(orig, fbs), center_frequency,
                                                    is_scalar, !prev_nonpar, eps, Sre, Sim);
                        double cr = std::sqrt(gmed), ci = 0.0;
                        if (res)
                            cr = Sre * std::sqrt(gmed), ci = Sim * std::sqrt(gmed);
                        rsu_scale(p_xprmatN, p_gainN, ii, n_rayN_t, cr, ci);
                        out_cur = (short)iM;   // current_out <- iM
                        out_prev = (short)cur; // prev_out <- old cur
                    }
                }
                else // buf != 0: ignore hit, continue in cur, swap buffer
                {
                    rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, dest)), is_scalar);
                    out_buf = (short)(same_materials(buf, M1) ? M2 : M1);
                }
            }
            else if (nH == 2 && typeH == 10)
            {
                // Edge o-i-o (Section 10.4, nH == 2). No S (graze, not a slab).
                if (cur == 0)
                {
                    // IG; current_out <- 0
                    out_cur = (short)0;
                }
                else if (same_materials(M1, M2)) // ignore hit
                {
                    rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, dest)), is_scalar);
                }
                else // i-i transition
                {
                    double g = MED(cur, D(orig, fbs)) * TRN(cur, M1) * MED(M1, D(fbs, dest));
                    rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, g, is_scalar);
                    out_cur = (short)M1;
                }
            }
            else if (nH == 2 && typeH == 11)
            {
                // Edge i-o-i (Section 10.5, nH == 2). No S, no flag (edge normals not a slab pair).
                if (cur == 0)
                {
                    // IG; current_out <- (d(fbs,sbs) > 1e-6 ? M2 : 0)
                    out_cur = (short)((D(fbs, sbs) > 1.0e-6) ? M2 : 0);
                }
                else if (same_materials(M1, M2)) // ignore hit
                {
                    rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, dest)), is_scalar);
                }
                else // i-i transition: MED(M2, d(fbs,dest) - off (unclamped))
                {
                    rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(M2, D(fbs, dest) - ray_offset), is_scalar);
                    out_cur = (short)M2;
                }
            }
            else if (nH > 2)
            {
                // Multi-hit (Section 10.1-10.5, nH > 2)
                if (cur == 0) // outside
                {
                    if (buf != 0) // cannot have i-i transition in buffer
                        rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
                    else if (typeH == 1 || typeH == 7) // o-i
                    {
                        // IG; current_out <- M1, +flag
                        out_cur = (short)M1;
                        bool nonpar = wedge_nonparallel();
                        out_prev = (short)(nonpar ? (int)0x8000 : 0);
                    }
                    else if (typeH == 2) // false inside: IG
                    {
                    }
                    else if (typeH == 10) // edge o-i-o, stay outside: IG
                    {
                    }
                    else if (typeH == 13) // edge o-i
                    {
                        // IG; current_out <- M1, buffer_out <- M2, +flag
                        out_cur = (short)M1;
                        out_buf = (short)M2;
                        bool nonpar = wedge_nonparallel();
                        out_prev = (short)(nonpar ? (int)0x8000 : 0);
                    }
                    else // some other hit type
                        rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
                }
                else // inside
                {
                    if (typeH == 1 || typeH == 7 || typeH == 13) // nested o-i, overlapping mesh
                    {
                        rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, fbs) + ray_offset), is_scalar);
                        out_buf = (short)M1;
                    }
                    else if (typeH == 2 || typeH == 14) // i-o
                    {
                        if (buf == 0) // cavity exit, IG * S
                        {
                            double Sre = 0.0, Sim = 0.0;
                            bool res = slab_airy_factor(mtl_prop, cur, 0, 0, theta, D(orig, fbs), center_frequency,
                                                        is_scalar, !prev_nonpar, eps, Sre, Sim);
                            if (res)
                                rsu_scale(p_xprmatN, p_gainN, ii, n_rayN_t, Sre, Sim);
                            out_cur = (short)0;
                        }
                        else if (same_materials(buf, M1)) // M2 embedded in M1
                        {
                            rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, fbs) + ray_offset), is_scalar);
                            out_buf = (short)0;
                        }
                        else
                        {
                            double g = MED(cur, D(orig, fbs)) * TRN(cur, buf) * MED(buf, ray_offset);
                            rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, g, is_scalar);
                            out_cur = (short)buf;
                            out_buf = (short)0;
                        }
                    }
                    else if (typeH == 4 || typeH == 5) // i-i
                    {
                        if (buf != 0) // spurious (probable false detection): IG
                        {
                            out_buf = (short)0;
                        }
                        else // cavity transition, IG * S
                        {
                            int iM = (typeH == 5) ? M2 : M1;
                            double Sre = 0.0, Sim = 0.0;
                            bool res = slab_airy_factor(mtl_prop, cur, iM, prev_mat, theta, D(orig, fbs), center_frequency,
                                                        is_scalar, !prev_nonpar, eps, Sre, Sim);
                            if (res)
                                rsu_scale(p_xprmatN, p_gainN, ii, n_rayN_t, Sre, Sim);
                            out_cur = (short)iM;   // current_out <- iM
                            out_prev = (short)cur; // prev_out <- old cur
                        }
                    }
                    else if (typeH == 8) // overlapping i-o
                    {
                        if (buf == 0) // cavity exit, IG * S
                        {
                            double Sre = 0.0, Sim = 0.0;
                            bool res = slab_airy_factor(mtl_prop, cur, 0, 0, theta, D(orig, fbs), center_frequency,
                                                        is_scalar, !prev_nonpar, eps, Sre, Sim);
                            if (res)
                                rsu_scale(p_xprmatN, p_gainN, ii, n_rayN_t, Sre, Sim);
                            out_cur = (short)0;
                        }
                        else
                        {
                            double g = MED(cur, D(orig, fbs)) * TRN(cur, 0);
                            rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, g, is_scalar);
                            out_cur = (short)0;
                            out_buf = (short)0;
                        }
                    }
                    else if (typeH == 10) // edge o-i-o (the cur == 0 guard at source 824-828 is dead)
                    {
                        if (buf == 0)
                        {
                            if (same_materials(M1, M2)) // ignore hit
                                rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, fbs) + ray_offset), is_scalar);
                            else // i-i transition
                            {
                                double g = MED(cur, D(orig, fbs)) * TRN(cur, M1) * MED(M1, ray_offset);
                                rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, g, is_scalar);
                                out_cur = (short)M1;
                            }
                        }
                        else // buf != 0: virtual i-i
                        {
                            if (same_materials(buf, M1))
                            {
                                rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, fbs) + ray_offset), is_scalar);
                                out_buf = (short)0;
                            }
                            else
                            {
                                double g = MED(cur, D(orig, fbs)) * TRN(cur, buf) * MED(buf, ray_offset);
                                rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, g, is_scalar);
                                out_cur = (short)buf;
                                out_buf = (short)0;
                            }
                        }
                    }
                    else if (typeH == 11) // edge i-o-i (the cur == 0 guard at source 865-871 is dead)
                    {
                        if (buf == 0)
                        {
                            if (same_materials(M1, M2)) // ignore hit
                                rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, MED(cur, D(orig, fbs) + ray_offset), is_scalar);
                            else // i-i transition
                            {
                                double g = MED(cur, D(orig, fbs)) * TRN(cur, M2) * MED(M2, ray_offset);
                                rsu_replace(p_xprmatN, p_gainN, ii, n_rayN_t, g, is_scalar);
                                out_cur = (short)M2;
                            }
                        }
                        else // buf != 0: spurious, IG
                        {
                            out_buf = (short)0;
                        }
                    }
                    else
                    {
                        // Unmatched inside type (TR or a degenerate out_type 0). The source leaves
                        // power unchanged (no-op pass-through); the unified global-default KILL
                        // replaces it (Section 10.0). Not exercised on the diffraction reference path.
                        rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
                    }
                }
            }
            else
            {
                // Global default: any unmatched (out_type, nH, state) -> KILL (Section 10.0).
                rsu_kill(p_xprmatN, p_gainN, ii, n_rayN_t);
            }
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