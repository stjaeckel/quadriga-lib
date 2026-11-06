// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_paths
Antenna-free paths of one link at one position

- Computes the antenna-free paths (gains, lengths, polarization matrices, optionally angles and scatterers) of the
  kept rays of `rays` at new Tx and Rx positions (default: the ray-data positions). Kept are LOS and GR rays always
  (also at zero power) and NLOS rays with power > 0 at some frequency; paths in ray order, `ray_ind` maps them to
  the rays (e.g. `rays.tap(ray_ind)`)
- `"phase"`: use the outputs with [[get_channels_planar]] and the arrays at the positions of `rays`; the displacement
  phase of TR 38.901 eq. (7.5-22) is in `M`. With the arrays at the new positions the LOS path is no longer
  identified and the displacement phase counts twice, without an error
- `"drift"`: use the outputs at the new positions with [[get_channels_spherical]] or [[get_channels_multifreq]], or
  pass them to [[qrt_paths_from_scatterers]]; needs the scatterers of .[[solve_scatterers]]

| Quantity | `"phase"` | `"drift"` |
| --- | --- | --- |
| Path length | Tx-Rx distance of `rays` + c·delay | Polyline Tx, FBS, LBS, Rx; LOS direct line; GR via the specular point |
| Angles | Ray data | Geometry: towards FBS and LBS, LOS direct line, GR towards the image points |
| `M` | Ray `M` times the displacement phase | Ray `M`; GR recomputed when an end moved |
| `fbs_pos`, `lbs_pos` | `[3, 0]` | Scatterers of the rays; LOS midpoint of the direct line; GR specular point |

- `"drift"` with an end moved (any coordinate differs from `rays`): GR power by the power law of `rays.method`
  (`(d3D / d_GR)²` for the TR methods, 1 for `"quadriga"`), GR `M` the scalar multiple of `diag(R_par, -R_perp)` (TR
  38.901 eqs. (7.6-38), (7.6-39)) that best fits the stored matrix, which replaces a matrix of another form (e.g. a
  conjugated one). Fallback rays have their longer polyline, also at the positions of `rays`
- Path gain per frequency: the coefficient power; with `kf_ratio` LOS and GR are rescaled against NLOS so that the
  K-factor of the coefficient powers changes by `kf_ratio`, LOS + NLOS keep their sum; times `10^(link_gain_dB / 10)`;
  with `oxygen` times the absorption of TR 38.901 clause 7.6.1 (Table 7.6.1-1 at each frequency) over each path's own
  length, without the delay offset τ_Δ of eq. (7.6-1) and at the carrier only (no in-band variation of eq. (7.6-2))
- Validity per moving end, with `b_eff` and `b_med` of [[gbsm_rays]]: `"phase"` up to about `sqrt(λ · b_eff / 8)`,
  typically 0.1 to 1 m; `"drift"` up to about `0.2 · b_med`, typically 1.5 to 12 m, new segment from `0.5 · b_med`;
  in `"drift"` each path length changes by at most the displacement. [[gbsm_single_link_channel]] reports the
  displacement relative to the limit
- Throws `std::invalid_argument` for invalid input; in `"drift"` also without scatterers, and for a GR ray at moved
  ends without permittivity in `rays` or with an end below z = 0

## Declaration:
```
void quadriga_lib::ssf_paths(
    const gbsm_rays &rays,
    const arma::vec &link_gain_dB,
    arma::mat &path_gain,
    arma::vec &path_length,
    arma::cube &M,
    const std::string &mobility = "auto",
    const arma::vec *tx_pos = nullptr,
    const arma::vec *rx_pos = nullptr,
    const arma::vec *kf_ratio = nullptr,
    arma::vec *aod = nullptr,
    arma::vec *eod = nullptr,
    arma::vec *aoa = nullptr,
    arma::vec *eoa = nullptr,
    arma::mat *fbs_pos = nullptr,
    arma::mat *lbs_pos = nullptr,
    arma::u32_vec *path_type = nullptr,
    arma::uvec *ray_ind = nullptr,
    bool oxygen = false);
```

## Inputs:
- **`rays`** — Ray data of the link
- **`link_gain_dB`** — [Link gain](cpp_api.html#sec-geometry-based-stochastic-model) at the position in [dB], per
  frequency; 0 for data from paths; `[n_freq]`
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"` (see Terms)
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx position in [m]; omitted: the positions of `rays`; `[3]`
- **`kf_ratio`** *(optional)* — KF(position) / KF(drop), >= 0, per frequency; omitted: 1; `[n_freq]`
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1

## Outputs:
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`M`** — Polarization transfer matrices, rows as in [[get_channels_planar]]; `[8, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** *(optional)* — Departure and arrival angles in [rad], GCS; `[n_path]`
- **`fbs_pos`**, **`lbs_pos`** *(optional)* — Scatterer positions in [m]; `[3, n_path]`, `[3, 0]` in `"phase"`
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS; `[n_path]`
- **`ray_ind`** *(optional)* — Index of the ray in `rays`, 0-based; `[n_path]`

## See also:
- [[gbsm_rays]] (input data, validity scales `b_eff` and `b_med`)
- [[gbsm_single_link_channel]] (channel coefficients of one link over a segment)
- [[gbsm_rays_from_paths]] (ray data from path data, e.g. of a QRT file)
- .[[solve_scatterers]] (scatterers for `"drift"`)
MD!*/

void quadriga_lib::ssf_paths(const gbsm_rays &rays,
                             const arma::vec &link_gain_dB,
                             arma::mat &path_gain,
                             arma::vec &path_length,
                             arma::cube &M,
                             const std::string &mobility,
                             const arma::vec *tx_pos,
                             const arma::vec *rx_pos,
                             const arma::vec *kf_ratio,
                             arma::vec *aod,
                             arma::vec *eod,
                             arma::vec *aoa,
                             arma::vec *eoa,
                             arma::mat *fbs_pos,
                             arma::mat *lbs_pos,
                             arma::u32_vec *path_type,
                             arma::uvec *ray_ind,
                             bool oxygen)
{
    // Validation; no output is touched before all checks passed
    const std::string err = rays.is_valid();
    if (!err.empty())
        throw std::invalid_argument(err);
    if (mobility != "phase" && mobility != "drift" && mobility != "auto")
        throw std::invalid_argument("Input 'mobility' must be 'phase', 'drift' or 'auto'.");
    const arma::uword n_ray = rays.n_ray(), n_freq = rays.n_freq();
    if (link_gain_dB.n_elem != n_freq)
        throw std::invalid_argument("Input 'link_gain_dB' must have n_freq elements.");
    if (!link_gain_dB.is_finite())
        throw std::invalid_argument("Input 'link_gain_dB' must be finite.");
    const std::pair<std::string, const arma::vec *> pos[2] = {{"tx_pos", tx_pos}, {"rx_pos", rx_pos}};
    for (const auto &p : pos)
        if (p.second != nullptr && p.second->n_elem != 3)
            throw std::invalid_argument("Input '" + p.first + "' must have 3 elements.");
        else if (p.second != nullptr && !p.second->is_finite())
            throw std::invalid_argument("Input '" + p.first + "' must be finite.");
    if (kf_ratio != nullptr && kf_ratio->n_elem != n_freq)
        throw std::invalid_argument("Input 'kf_ratio' must have n_freq elements.");
    if (kf_ratio != nullptr && (!kf_ratio->is_finite() || arma::any(*kf_ratio < 0.0)))
        throw std::invalid_argument("Input 'kf_ratio' must be finite and >= 0.");

    const bool drift = mobility == "drift" || (mobility == "auto" && rays.method == "quadriga");
    const double *T0 = rays.tx_pos.memptr(), *R0 = rays.rx_pos.memptr();
    const double *T = (tx_pos != nullptr) ? tx_pos->memptr() : T0, *R = (rx_pos != nullptr) ? rx_pos->memptr() : R0;
    const bool moved = T[0] != T0[0] || T[1] != T0[1] || T[2] != T0[2] || R[0] != R0[0] || R[1] != R0[1] ||
                       R[2] != R0[2]; // exact: the positions of rays reproduce the drop geometry
    const bool gr_new = drift && moved && arma::any(rays.path_type == 1);
    if (drift && n_ray > 0 && rays.fbs_pos.n_cols != n_ray)
        throw std::invalid_argument("Mobility 'drift' needs 'rays.fbs_pos' and 'rays.lbs_pos'; solve the scatterers "
                                    "first.");
    if (gr_new && rays.gr_eps_re.empty())
        throw std::invalid_argument("Input 'rays.gr_eps_re' is required to recompute the ground reflection away from "
                                    "the positions of 'rays'.");
    if (gr_new && (T[2] < 0.0 || R[2] < 0.0))
        throw std::invalid_argument("Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'.");

    // Kept rays: LOS and GR always, NLOS with power > 0 at some frequency
    std::vector<arma::uword> keep;
    for (arma::uword i = 0; i < n_ray; ++i)
        if (rays.path_type(i) != 2 || arma::any(rays.power.row(i) > 0.0))
            keep.push_back(i);
    const arma::uword n_path = (arma::uword)keep.size(), n_sc = drift ? n_path : 0;

    // Outputs: resized only when the size differs, so that pre-sized outputs are written in place
    if (path_gain.n_rows != n_path || path_gain.n_cols != n_freq)
        path_gain.set_size(n_path, n_freq);
    if (path_length.n_elem != n_path)
        path_length.set_size(n_path);
    if (M.n_rows != 8 || M.n_cols != n_path || M.n_slices != n_freq)
        M.set_size(8, n_path, n_freq);
    for (arma::vec *v : {aod, eod, aoa, eoa})
        if (v != nullptr && v->n_elem != n_path)
            v->set_size(n_path);
    for (arma::mat *m : {fbs_pos, lbs_pos})
        if (m != nullptr && (m->n_rows != 3 || m->n_cols != n_sc))
            m->set_size(3, n_sc);
    if (path_type != nullptr && path_type->n_elem != n_path)
        path_type->set_size(n_path);
    if (ray_ind != nullptr && ray_ind->n_elem != n_path)
        ray_ind->set_size(n_path);

    const double c0 = 299792458.0, pi2 = 6.283185307179586;

    // K rescale per frequency (TR coefficient powers with K' = kf_ratio·K_c, K_c = S_L / S_N the K-factor of the
    // coefficient powers): a_L = r·(S_L + S_N) / (r·S_L + S_N) for LOS and GR, a_N = (S_L + S_N) / (r·S_L + S_N)
    // for NLOS, so that LOS + NLOS keep their sum; r·S_L + S_N = 0 gives a_L = 0, a_N = 1
    arma::vec a_L(n_freq, arma::fill::ones), a_N(n_freq, arma::fill::ones), lin(n_freq), alpha(n_freq);
    for (arma::uword k = 0; k < n_freq; ++k)
    {
        double S_L = 0.0, S_N = 0.0;
        for (arma::uword i = 0; i < n_ray; ++i)
            (rays.path_type(i) == 0 ? S_L : S_N) += (rays.path_type(i) == 1) ? 0.0 : rays.power(i, k);
        const double r = (kf_ratio != nullptr) ? (*kf_ratio)(k) : 1.0, q = r * S_L + S_N;
        if (S_L + S_N > 0.0)
            a_L(k) = (q > 0.0) ? r * (S_L + S_N) / q : 0.0, a_N(k) = (q > 0.0) ? (S_L + S_N) / q : 1.0;
        lin(k) = std::pow(10.0, 0.1 * link_gain_dB(k));

        // Oxygen absorption in [dB/km], TR 38.901 Table 7.6.1-1 at 52, 53, ..., 68 GHz, linear interpolation, 0
        // outside; τ_Δ of eq. (7.6-1) is not added: each path's own length is used
        const double tab[17] = {0.0, 1.0, 2.2, 4.0, 6.6, 9.7, 12.6, 14.6, 15.0,
                                14.6, 14.3, 10.5, 6.8, 3.9, 1.9, 1.0, 0.0};
        const double f = rays.center_frequency(k) * 1e-9 - 52.0;
        alpha(k) = 0.0;
        if (oxygen && f > 0.0 && f < 16.0) // index only inside the table: the cast of a larger f overflows int
        {
            const int j = (int)f;
            alpha(k) = tab[j] + (f - (double)j) * (tab[j + 1] - tab[j]);
        }
    }

    // Ground reflection at the ray-data and the new positions: specular point G, Fresnel coefficients and power law
    // g = (d3D / d_GR')², d_GR' = max(d_GR, 1e-5), for the TR methods, 1 for quadriga
    double G[3];
    qd_gbsm_specular_point(T, R, G);
    const double mid[3] = {0.5 * (T[0] + R[0]), 0.5 * (T[1] + R[1]), 0.5 * (T[2] + R[2])};
    const double R_img[3] = {R[0], R[1], -R[2]}, T_img[3] = {T[0], T[1], -T[2]};
    auto gr_law = [&](const double *a, const double *b)
    {
        const double d3D = qd_gbsm_dist(a, b), d2D = std::hypot(b[0] - a[0], b[1] - a[1]);
        const double d_gr = std::max(std::sqrt(d2D * d2D + (a[2] + b[2]) * (a[2] + b[2])), 1e-5);
        return (rays.method == "quadriga") ? 1.0 : (d3D / d_gr) * (d3D / d_gr);
    };
    const double g_ratio = gr_new && gr_law(T0, R0) > 0.0 ? gr_law(T, R) / gr_law(T0, R0) : 1.0;

    const double dT[3] = {T[0] - T0[0], T[1] - T0[1], T[2] - T0[2]}, dR[3] = {R[0] - R0[0], R[1] - R0[1], R[2] - R0[2]};
    const double d3D_0 = qd_gbsm_dist(T0, R0);
    for (arma::uword p = 0; p < n_path; ++p)
    {
        const arma::uword i = keep[p];
        const unsigned type = rays.path_type(i);
        double L = d3D_0 + c0 * rays.delay(i), ang[4] = {rays.aod(i), rays.eod(i), rays.aoa(i), rays.eoa(i)};
        double proj = 0.0;                 // "phase": r_rx·dR + r_tx·dT
        const double *F = mid, *B = mid; // "drift": scatterers of the path
        if (!drift)
        {
            const double ce = std::cos(ang[1]), ca = std::cos(ang[3]);
            proj = ce * std::cos(ang[0]) * dT[0] + ce * std::sin(ang[0]) * dT[1] + std::sin(ang[1]) * dT[2] +
                   ca * std::cos(ang[2]) * dR[0] + ca * std::sin(ang[2]) * dR[1] + std::sin(ang[3]) * dR[2];
        }
        else if (type == 2)
        {
            F = rays.fbs_pos.colptr(i), B = rays.lbs_pos.colptr(i);
            L = qd_gbsm_dist(T, F) + qd_gbsm_dist(F, B) + qd_gbsm_dist(B, R);
            qd_gbsm_direction(T, {F, B, R}, ang[0], ang[1]), qd_gbsm_direction(R, {B, F, T}, ang[2], ang[3]);
        }
        else if (type == 0)
        {
            L = qd_gbsm_dist(T, R);
            qd_gbsm_direction(T, {R}, ang[0], ang[1]), qd_gbsm_direction(R, {T}, ang[2], ang[3]);
        }
        else // GR: angles towards the image points, also defined for an end on the ground
        {
            F = G, B = G, L = qd_gbsm_dist(T, G) + qd_gbsm_dist(G, R);
            qd_gbsm_direction(T, {R_img}, ang[0], ang[1]), qd_gbsm_direction(R, {T_img}, ang[2], ang[3]);
        }

        for (arma::uword k = 0; k < n_freq; ++k)
        {
            const double *m0 = rays.M.slice_memptr(k) + 8 * i;
            double *m = M.slice_memptr(k) + 8 * p, P = rays.power(i, k);
            for (int e = 0; e < 8; ++e)
                m[e] = m0[e];
            if (!drift) // displacement phase exp(j·2·pi·f/c·(r_rx·dR + r_tx·dT)), TR eq. (7.5-22)
            {
                const double ph = pi2 * rays.center_frequency(k) / c0 * proj, cp = std::cos(ph), sp = std::sin(ph);
                for (int e = 0; e < 8; e += 2)
                    m[e] = m0[e] * cp - m0[e + 1] * sp, m[e + 1] = m0[e] * sp + m0[e + 1] * cp;
            }
            else if (type == 1 && gr_new)
            {
                // M(t) = q·diag(R_par(t), -R_perp(t)) with q = <D_0, M_0> / |D_0|²_F the scalar that best maps
                // D_0 = diag(R_par, -R_perp) at the ray-data positions to the stored M_0 (<X, Y> = sum conj(X)·Y);
                // exact at the ray-data positions and continuous for any complex scalar multiple; D_0 = 0 keeps M_0
                std::complex<double> par0, per0, par, per;
                const double e_re = rays.gr_eps_re(k), e_im = rays.gr_eps_im(k);
                qd_gbsm_fresnel(e_re, e_im, T0[2], R0[2], std::hypot(R0[0] - T0[0], R0[1] - T0[1]), par0, per0);
                qd_gbsm_fresnel(e_re, e_im, T[2], R[2], std::hypot(R[0] - T[0], R[1] - T[1]), par, per);
                const double nD = std::norm(par0) + std::norm(per0);
                if (nD > 0.0)
                {
                    const std::complex<double> q = (std::conj(par0) * std::complex<double>(m0[0], m0[1]) -
                                                    std::conj(per0) * std::complex<double>(m0[6], m0[7])) /
                                                   nD;
                    const std::complex<double> a = q * par, b = -q * per;
                    m[0] = a.real(), m[1] = a.imag(), m[2] = m[3] = m[4] = m[5] = 0.0, m[6] = b.real(), m[7] = b.imag();
                }
                P *= g_ratio; // GR power law of the method
            }
            P *= (type == 2 ? a_N(k) : a_L(k)) * lin(k);
            path_gain(p, k) = (alpha(k) > 0.0) ? P * std::pow(10.0, -alpha(k) * L * 1e-4) : P;
        }

        path_length(p) = L;
        double *out[4] = {aod ? aod->memptr() : nullptr, eod ? eod->memptr() : nullptr, aoa ? aoa->memptr() : nullptr,
                          eoa ? eoa->memptr() : nullptr};
        for (int q = 0; q < 4; ++q)
            if (out[q] != nullptr)
                out[q][p] = ang[q];
        for (int j = 0; j < 3 && drift; ++j)
        {
            if (fbs_pos != nullptr)
                (*fbs_pos)(j, p) = F[j];
            if (lbs_pos != nullptr)
                (*lbs_pos)(j, p) = B[j];
        }
        if (path_type != nullptr)
            (*path_type)(p) = type;
        if (ray_ind != nullptr)
            (*ray_ind)(p) = i;
    }
}
