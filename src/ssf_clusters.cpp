// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    const double qd_pi = 3.14159265358979323846;

    // Azimuth wrap to [-pi, pi)
    inline double qd_wrap(double x) { return x - 2.0 * qd_pi * std::floor((x + qd_pi) / (2.0 * qd_pi)); }

    // Zenith angle in [deg] to elevation in [rad]: wrapped to [0, 360), [180, 360) -> 360 - t (TR 38.901 clause 7.5
    // step 7), elevation = 90 - zenith
    inline double qd_elev(double t)
    {
        t -= 360.0 * std::floor(t / 360.0);
        return (90.0 - (t > 180.0 ? 360.0 - t : t)) * qd_pi / 180.0;
    }

    // Clusters of one link by TR 38.901 clause 7.5 steps 5 to 7, one frequency (method "38.901")
    // - x: values of the shared set of the link, column b·N + n; z: shadowing values, [N]; N_l active clusters
    // - ds in [s]; kf linear (0 in NLOS); as = {ASD, ASA, ZSD, ZSA} in [deg]
    // - centre = {AOD, AOA, ZOD, ZOA} in [deg]: LOS direction (LOS) or the mean angles of eqs. (7.5-11), (7.5-16),
    //   (7.5-19) (NLOS)
    // - Outputs for n < N_l: delay [s] (relative, without abs TOA), coefficient power, angles [rad] in
    //   ang[a·N_l + n] for a = 0 AOD, 1 AOA, 2 EOD, 3 EOA
    void qd_ssf_tr(const double *x, const double *z, arma::uword N, arma::uword N_l, bool los, double r_tau,
                   double zeta, double ds, double kf, const double *as, const double *centre, double *delay,
                   double *power, double *ang)
    {
        // Step 5: delays, eqs. (7.5-1) to (7.5-4); no sort, the first cluster has the smallest initial delay
        std::vector<double> tau(N_l), lam(N_l), P(N_l), Pa(N_l), v(N_l);
        arma::uword f = 0;
        for (arma::uword n = 0; n < N_l; ++n)
        {
            tau[n] = -r_tau * ds * std::log(std::max(x[n], DBL_MIN));
            f = (tau[n] < tau[f]) ? n : f;
        }
        const double t_min = tau[f];
        const double k_db = std::max(10.0 * std::log10(kf), -5.0); // K limited to -5 dB in the polynomials
        const double k2 = k_db * k_db, k3 = k2 * k_db;
        const double c_tau = 0.7705 - 0.0433 * k_db + 0.0002 * k2 + 0.000017 * k3;

        // Step 6: powers, eqs. (7.5-5) to (7.5-8), normalised in the log domain
        double l_max = -INFINITY;
        for (arma::uword n = 0; n < N_l; ++n)
        {
            tau[n] -= t_min;
            delay[n] = los ? tau[n] / c_tau : tau[n];
            lam[n] = -tau[n] * (r_tau - 1.0) / (r_tau * ds) - zeta * z[n] * std::log(10.0) / 10.0;
            l_max = std::max(l_max, lam[n]);
        }
        double sum = 0.0, p_max = 0.0, pa_max = 0.0;
        for (arma::uword n = 0; n < N_l; ++n)
            P[n] = std::exp(lam[n] - l_max), sum += P[n];
        for (arma::uword n = 0; n < N_l; ++n)
            P[n] /= sum, p_max = std::max(p_max, P[n]);
        const double k1 = 1.0 / (kf + 1.0), kk = kf / (kf + 1.0);
        for (arma::uword n = 0; n < N_l; ++n)
        {
            Pa[n] = los ? P[n] * k1 + (n == f ? kk : 0.0) : P[n];
            pa_max = std::max(pa_max, Pa[n]);
            power[n] = (P[n] < std::pow(10.0, -2.5) * p_max) ? 0.0 : P[n] * k1; // -25 dB removal, no renormalisation
        }

        // Step 7: angles, eqs. (7.5-9) to (7.5-19), in degrees
        const double c_phi_n[14] = {0.779, 0.860, 0.921, 0.973, 1.018, 1.090, 1.123,
                                    1.146, 1.190, 1.211, 1.226, 1.273, 1.289, 1.358};
        const double n_phi[14] = {4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 19, 20, 25};
        const double c_theta_n[12] = {0.788, 0.847, 0.889, 0.957, 1.031, 1.104,
                                      1.1072, 1.1088, 1.1276, 1.184, 1.178, 1.282};
        const double n_theta[12] = {6, 7, 8, 10, 11, 12, 14, 15, 16, 19, 20, 25};
        const double c_phi = qd_interp_linear(c_phi_n, n_phi, 14, (double)N_l) *
                             (los ? 1.1035 - 0.028 * k_db - 0.002 * k2 + 0.0001 * k3 : 1.0);
        const double c_theta = qd_interp_linear(c_theta_n, n_theta, 12, (double)N_l) *
                               (los ? 1.3086 + 0.0339 * k_db - 0.0077 * k2 + 0.0002 * k3 : 1.0);
        for (arma::uword a = 0; a < 4; ++a)
        {
            for (arma::uword n = 0; n < N_l; ++n)
            {
                const double rho = std::log(std::max(Pa[n], DBL_MIN) / pa_max);
                const double a1 = (a < 2) ? 2.0 * (as[a] / 1.4) * std::sqrt(-rho) / c_phi : -as[a] * rho / c_theta;
                v[n] = (x[(1 + a) * N + n] >= 0.0 ? a1 : -a1) + x[(5 + a) * N + n] * as[a] / 7.0;
            }
            const double v_f = v[f];
            for (arma::uword n = 0; n < N_l; ++n)
            {
                const double t = centre[a] + (los ? v[n] - v_f : v[n]); // eqs. (7.5-12), (7.5-17) align cluster f
                ang[a * N_l + n] = (a < 2) ? qd_wrap(t * qd_pi / 180.0) : qd_elev(t);
            }
        }
    }

    // Clusters of one link by TR 38.901 clause 7.6.5.1 steps 5' to 7', all frequencies (method "38.901_MF")
    // - x: values of the shared set of the link, column b·N + n; z: shadowing values, z[k·N + n]; N_l active clusters
    // - sp0 = {DS0, ASD0, ASA0, ZSD0, ZSA0} anchor spreads; sp[k·5 + i] the same at frequency k; kf[k] linear K
    //   (0 in NLOS); DS in [s], angular spreads in [deg]
    // - centre = {AOD, AOA, ZOD, ZOA} of the LOS direction in [deg]
    // - Outputs for n < N_l: delay[k·N_l + n] [s] (relative, without abs TOA), power[k·N_l + n] coefficient power,
    //   ang[(k·4 + a)·N_l + n] [rad] for a = 0 AOD, 1 AOA, 2 EOD, 3 EOA
    void qd_ssf_mf(const double *x, const double *z, arma::uword N, arma::uword N_l, arma::uword F, bool los,
                   double zeta, const double *sp0, const double *sp, const double *kf, const double *centre,
                   double *delay, double *power, double *ang)
    {
        const double r = 1.5, s2 = std::sqrt(2.0);
        auto wrap180 = [](double v) { return v - 360.0 * std::floor((v + 180.0) / 360.0); };

        // Step 5': nominal delays and angles; arg(exp(-j·v)) = wrap180(-v); the Laplacian term of the TR,
        // sgn(V - 0.5)·ln(1 - 2·abs(V - 0.5)) with V = Phi(y), as sgn(y)·ln(erfc(abs(y)/sqrt(2)))
        std::vector<double> tau(N_l), a0(4 * N_l), lam(N_l);
        for (arma::uword n = 0; n < N_l; ++n)
        {
            const bool first = los && n == 0; // TR cluster 1 in LOS: delay and angles 0
            tau[n] = first ? 0.0 : -r * sp0[0] * std::log(std::max(x[n], DBL_MIN));
            for (arma::uword a = 0; a < 4; ++a)
            {
                const double y = x[(5 + a) * N + n];
                const double v = (a < 2) ? y : (double)((y > 0.0) - (y < 0.0)) *
                                                   std::log(std::max(std::erfc(std::abs(y) / s2), DBL_MIN)) / s2;
                a0[a * N_l + n] = first ? 0.0 : wrap180(-r * sp0[1 + a] * v);
            }
        }

        for (arma::uword k = 0; k < F; ++k)
        {
            // Step 6': eqs. (7.6.30a) to (7.6.30h), normalised in the log domain; the LOS power is in slot 0
            const double *s = sp + 5 * k;
            double g[5];
            g[0] = std::max(r * sp0[0] - s[0], 0.0) / (s[0] * r * sp0[0]);
            for (arma::uword i = 1; i < 5; ++i)
                g[i] = (i < 3) ? std::sqrt(std::max(r * sp0[i] * r * sp0[i] - s[i] * s[i], 0.0)) / (s[i] * r * sp0[i])
                               : std::max(r * sp0[i] - s[i], 0.0) / (s[i] * r * sp0[i]);
            double l_max = -INFINITY, sum = 0.0;
            for (arma::uword n = 0; n < N_l; ++n)
            {
                const double e1 = a0[n] * g[1] / s2, e2 = a0[N_l + n] * g[2] / s2;
                lam[n] = -tau[n] * g[0] - e1 * e1 - e2 * e2 - s2 * std::abs(a0[2 * N_l + n]) * g[3] -
                         s2 * std::abs(a0[3 * N_l + n]) * g[4] - zeta * z[k * N + n] * std::log(10.0) / 10.0;
                l_max = std::max(l_max, lam[n]);
            }
            for (arma::uword n = 0; n < N_l; ++n)
                power[k * N_l + n] = std::exp(lam[n] - l_max), sum += power[k * N_l + n];
            for (arma::uword n = 0; n < N_l; ++n)
                power[k * N_l + n] /= sum * (kf[k] + 1.0);

            // Step 7': eqs. (7.6.30i) to (7.6.30m), cluster level; the wrapped nominal angles are scaled in LOS
            const double c_tau = std::sqrt(1.0 + kf[k] / 2.0), c_ang = std::sqrt(1.0 + kf[k]);
            for (arma::uword n = 0; n < N_l; ++n)
            {
                delay[k * N_l + n] = c_tau * tau[n];
                for (arma::uword a = 0; a < 4; ++a)
                {
                    const double t = centre[a] + c_ang * a0[a * N_l + n];
                    ang[(k * 4 + a) * N_l + n] = (a < 2) ? qd_wrap(t * qd_pi / 180.0) : qd_elev(t);
                }
            }
        }
    }

    // Clusters of one link by the QuaDRiGa method (paper: Jaeckel et al., 2019, section II), all frequencies
    // (method "quadriga")
    // - x: values of the shared set of the link, column b·N + n; N_l active clusters; F frequencies
    // - sp[k·5 + i] = {DS [s], ASD, ASA, ESD, ESA [rad]} at frequency k; kf[k] linear K (0 in NLOS)
    // - los_ang = {AOD, AOA, EOD, EOA} [rad]; gr: ground reflection with delay tau_gr [s], elevation e_gr [rad] and
    //   reflection power r2[k]; delta: EoD offset [rad]; toa: absolute TOA [s], the delays are shifted if > 0
    // - Outputs for n < N_l: delay[n] [s], power[k·N_l + n] coefficient power, ang[a·N_l + n] [rad] for a = 0 AOD,
    //   1 AOA, 2 EOD, 3 EOA
    void qd_ssf_quadriga(const double *x, arma::uword N, arma::uword N_l, arma::uword F, bool los, double r_tau,
                         const double *sp, const double *kf, const double *los_ang, bool gr, double tau_gr,
                         double e_gr, const double *r2, double delta, double toa, double *delay, double *power,
                         double *ang)
    {
        auto efold = [](double e) // elevation mirrored at the poles
        {
            e = qd_wrap(e);
            return e > 0.5 * qd_pi ? qd_pi - e : (e < -0.5 * qd_pi ? -qd_pi - e : e);
        };

        // Initial delays and angles, paper eqs. (1), (4), (5); (Phi(y) - 0.5)·pi = erf(y/sqrt(2))·pi/2
        std::vector<double> tt(N_l), at(4 * N_l), lam(N_l), P(N_l), A(N_l + 2), wt(N_l + 2);
        for (arma::uword n = 0; n < N_l; ++n)
        {
            tt[n] = -std::log(std::max(x[n], DBL_MIN));
            for (arma::uword a = 0; a < 4; ++a)
                at[a * N_l + n] = 0.5 * qd_pi * std::erf(x[(5 + a) * N + n] / std::sqrt(2.0));
        }

        // Centres: LOS azimuths; elevations weighted between the LOS and the ground reflection (weight 0 in NLOS)
        double w = 0.0, mx[5], mn[5];
        for (arma::uword k = 0; k < F && gr && los; ++k)
            w += r2[k] / (double)F;
        const double cen[4] = {los_ang[0], los_ang[1], (1.0 - w) * los_ang[2] + w * e_gr,
                               (1.0 - w) * los_ang[3] + w * e_gr};
        for (arma::uword i = 0; i < 5; ++i)
        {
            mx[i] = sp[i], mn[i] = sp[i];
            for (arma::uword k = 1; k < F; ++k)
                mx[i] = std::max(mx[i], sp[5 * k + i]), mn[i] = std::min(mn[i], sp[5 * k + i]);
        }

        const arma::uword n0 = gr ? 2 : 1; // fixed slots (LOS, ground reflection) in the scaling sets
        double s_tau = 0.0, s_ang[4] = {0.0, 0.0, 0.0, 0.0};
        for (arma::uword k = 0; k < F; ++k)
        {
            // Power coefficients, paper eqs. (7) to (11)
            const double *s = sp + 5 * k;
            double g[5] = {r_tau - 1.0, 0.56, 0.56, 0.76, 0.76};
            if (F > 1)
            {
                if (std::abs(mx[0] / mn[0] - 1.0) >= 1e-6)
                    g[0] = -1.5 * std::log(1.2 * std::min(std::max(s[0] / (mx[0] + mn[0]), 0.15), 0.85) - 0.15);
                for (arma::uword i = 1; i < 5; ++i)
                {
                    const double b = std::max(0.75 * s[i] / mx[i], 0.25);
                    g[i] = (i < 3) ? -2.2 * std::log(1.5 * b - 0.35) : -3.4 * std::log(1.2 * b - 0.1);
                }
            }

            // Scaling powers, paper eqs. (6), (12), normalised in the log domain; the LOS power split into LOS and
            // ground reflection
            double l_max = -INFINITY, sum = 0.0, e = 0.0, c = 0.0;
            for (arma::uword n = 0; n < N_l; ++n)
            {
                lam[n] = -tt[n] * g[0] - at[n] * at[n] * g[1] - at[N_l + n] * at[N_l + n] * g[2] -
                         std::abs(at[2 * N_l + n]) * g[3] - std::abs(at[3 * N_l + n]) * g[4];
                l_max = std::max(l_max, lam[n]);
            }
            for (arma::uword n = 0; n < N_l; ++n)
                P[n] = std::exp(lam[n] - l_max), sum += P[n];
            const double kk = kf[k] / (1.0 + kf[k]), s_los = kk * (1.0 - (gr ? r2[k] : 0.0)), s_gr = gr ? kk * r2[k] : 0.0;
            for (arma::uword n = 0; n < N_l; ++n)
            {
                P[n] /= sum * (1.0 + kf[k]), power[k * N_l + n] = P[n];
                e += P[n] * tt[n], c += P[n] * tt[n] * tt[n];
            }

            // Delay scale, paper eqs. (13), (14), in closed form with the ground reflection as a fixed path
            const double den = c - e * e, a2 = s[0] * s[0], B = s_gr * tau_gr * tau_gr, D = s_gr * tau_gr;
            const double t = std::max(a2 * c - a2 * e * e - B * c + B * e * e + c * D * D, 0.0);
            s_tau += ((den > 0.0) ? (gr ? (std::sqrt(t) + D * e) / den : s[0] / std::sqrt(den)) : 1.0) / (double)F;

            // Angle scales: the input spread over the wrapped spread of the initial angles, then iterated on the
            // linear spread of the scaled cluster offsets with the fixed offsets of LOS and ground reflection
            for (arma::uword a = 0; a < 4; ++a)
            {
                wt[0] = s_los, A[0] = (a < 2) ? 0.0 : los_ang[a] - cen[a];
                wt[1] = s_gr, A[1] = (a < 2) ? 0.0 : e_gr - cen[a];
                for (arma::uword n = 0; n < N_l; ++n)
                    wt[n0 + n] = P[n], A[n0 + n] = at[a * N_l + n];
                auto spread = [&](double mu, double f, bool wrapped)
                {
                    double m1 = 0.0, m2 = 0.0;
                    for (arma::uword i = 0; i < n0 + N_l; ++i)
                    {
                        const double v = wrapped ? qd_wrap(A[i] - mu) : (i < n0 ? A[i] : f * A[i]);
                        m1 += wt[i] * v, m2 += wt[i] * v * v;
                    }
                    return std::sqrt(std::max(m2 - m1 * m1, 0.0));
                };
                double ss = 0.0, cs = 0.0, sc = 1.0;
                for (arma::uword i = 0; i < n0 + N_l; ++i)
                    ss += wt[i] * std::sin(A[i]), cs += wt[i] * std::cos(A[i]);
                const double S0 = spread(std::atan2(ss, cs), 1.0, true);
                if (S0 > 0.0)
                {
                    double sL = 10.0 * std::log10(s[1 + a] / S0);
                    for (int i = 0; i < 9; ++i)
                    {
                        const double S1 = spread(0.0, std::pow(10.0, 0.1 * sL), false);
                        if (!(S1 > 0.0))
                            break;
                        const double sN = 10.0 * std::log10(s[1 + a] / S1);
                        if (std::abs(sN) <= 1e-4)
                            break;
                        sL += sN;
                    }
                    sc = std::pow(10.0, 0.1 * sL);
                }
                s_ang[a] += sc / (double)F;
            }
        }

        // Scaled offsets (caps of paper eq. (18)), the LOS direction added, elevations mirrored at the poles
        for (arma::uword a = 0; a < 4; ++a)
        {
            const double s = std::min(s_ang[a], (a < 2) ? 3.0 : 1.5);
            for (arma::uword n = 0; n < N_l; ++n)
            {
                const double b = s * at[a * N_l + n];
                ang[a * N_l + n] = (a < 2) ? qd_wrap(cen[a] + b) : efold(cen[a] + (a == 2 ? delta : 0.0) + efold(b));
            }
        }

        // Delays; absolute TOA: the first NLOS cluster at toa
        double t_min = INFINITY;
        for (arma::uword n = 0; n < N_l; ++n)
            delay[n] = s_tau * tt[n], t_min = std::min(t_min, delay[n]);
        for (arma::uword n = 0; n < N_l && toa > 0.0; ++n)
            delay[n] = delay[n] - t_min + toa;
    }
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_clusters
Cluster delays, powers and angles of a set of links, TR 38.901 clause 7.5 steps 5 to 7

- Turns the large-scale parameters of [[lsp_generate]] and the random values of [[ssf_draw]] into the delays, powers
  and angles of the clusters of each link, by one of three methods:
- `"38.901"` (default): TR 38.901 clause 7.5 steps 5 to 7 (exponential delays, powers with per-cluster shadowing,
  inverse Gaussian and Laplacian angle mapping with the scaling factors C_phi and C_theta); one frequency only
- `"38.901_MF"`: the multi-frequency method of TR 38.901 clause 7.6.5.1 (nominal delays and angles from anchor
  spreads, frequency-dependent powers, scaling by the K-factor per frequency)
- `"quadriga"`: the cluster generation of QuaDRiGa (paper: Jaeckel, Raschkowski, Burkhardt, Thiele, "A Spatially
  Consistent Geometric D2D Small-Scale Fading Model for Multiple Frequencies", 2019): continuous in space,
  frequency-independent delays and angles for several frequencies, swap-reciprocal, exact delay spread
- Slots per link: slot 0 = LOS path (delay 0, geometric LOS angles); slot 1 = ground reflection if `GR_enabled` is
  > 0; then the `N = NumClusters - 1` NLOS clusters, slot `1 + G + n` for cluster `n` (`G = 1` with ground
  reflection, else 0); `n_slot = NumClusters + G`. Slot `1 + G + n` always belongs to column `n` of each block of
  the random values (the clusters are not sorted)
- LOS condition: the scenario is a LOS scenario if `KF_mu > -30` (dB). In an NLOS scenario the LOS slot and the
  ground-reflection slot have power 0 whatever K-factor `lsp` holds, and every method uses its NLOS formulas.
  Legacy QuaDRiGa "NLOS" files with `KF_mu > -30` (e.g. `BERLIN_UMa_NLOS`, `QuaDRiGa_Industrial_NLOS`) therefore
  run the LOS formulas
- Units and conventions: delays in [s] relative to the LOS delay (absolute delay = d3D / c + delay); angles in
  [rad] in global coordinates, elevation convention (elevation = 90 deg - zenith); azimuths in [-pi, pi),
  elevations in [-pi/2, pi/2]; the LOS angles from the positions: AoD = atan2(dy, dx), EoD = atan2(dz, d2D),
  AoA = AoD + pi (wrapped), EoA = -EoD
- Powers are the powers of the channel coefficients of TR 38.901 eq. (7.5-30): LOS slot `K/(K+1)`, NLOS clusters
  `P_n/(K+1)` (sum `1/(K+1)`); in NLOS scenarios K = 0. Path loss and shadow fading are not included (step 12,
  the caller)
- Per-link number of clusters (TR 38.901 clause 7.6.15): `n_cluster(l)` NLOS clusters are active; the other slots
  get delay 0, power 0 and the LOS angles. Every normalisation, minimum, maximum and scaling runs over the active
  clusters; in `"38.901"` the scaling factors C_phi, C_theta use `n_cluster(l)`. `n_active(l)` counts the active
  clusters with a power > 0 at one frequency at least (removed clusters are not counted)
- Random values, `N` columns per block (see [[ssf_draw]]): the shared set `rnd` (block 0 delays, blocks 1 to 4
  signs of AOD, AOA, ZOD, ZOA, blocks 5 to 8 angle offsets of AOD, AOA, ZOD, ZOA) and one per-frequency set (the
  cluster shadowing) per frequency in `rnd_freq`. `"38.901"` uses all nine blocks and the shadowing;
  `"38.901_MF"` uses blocks 0, 5, 6, 7, 8 and the shadowing per frequency; `"quadriga"` uses blocks 0, 5, 6, 7, 8.
  Normal blocks are mapped to Uniform(0,1) by `0.5·erfc(-x/sqrt(2))` where a method needs uniform values
- Spatial consistency with `"38.901"` (TR 38.901 clause 7.6.3.1): the caller evaluates the sign and offset blocks
  (1 to 8) at the drop or segment-start positions and reuses these values along the track; the delays and the
  shadowing follow the positions. The method still jumps along tracks (see the table below)
- Inputs used per method (inputs that a method does not use are ignored and not validated):

| Input | `38.901` | `38.901_MF` | `quadriga` |
| --- | --- | --- | --- |
| `rnd` blocks | 0 to 8 | 0, 5, 6, 7, 8 | 0, 5, 6, 7, 8 |
| `rnd_freq` | required (one frequency) | required | ignored |
| `lsp_anchor` | ignored | required | ignored |
| `eod_offset_deg` | NLOS scenarios: mu_offset,ZOD = -value | ignored | NLOS scenarios: mean over the frequencies |
| `abs_toa` | NLOS scenarios: added | NLOS scenarios: added | NLOS scenarios: shift to the offset if > 0 |
| `gr_eps_re`, `gr_eps_im` | TR 38.901 clause 7.6.8 | TR 38.901 clause 7.6.8 | coefficient powers; power split inside the scaling |
| `o2i` | NLOS: mean ZOA 90 deg | ignored | ignored |
| `n_cluster` | yes, also in C_phi, C_theta | yes | yes |

- Absolute time of arrival (TR 38.901 clause 7.6.9), NLOS scenarios only: `"38.901"` and `"38.901_MF"` add
  `abs_toa(l)` to every active NLOS delay (eq. (7.6-43); with `"38.901"` the first cluster is then at
  `abs_toa(l)`); `"quadriga"` shifts the NLOS delays so that the first starts at `abs_toa(l)` where it is > 0. In
  LOS scenarios `abs_toa` is ignored (eq. (7.6-44) has no offset)
- Ground reflection (TR 38.901 clause 7.6.8), on if `GR_enabled > 0`; the ground is the plane z = 0 (`tx_pos` and
  `rx_pos` must have z >= 0). Slot 1: delay `(d_GR - d3D)/c`, computed as `4·h_t·h_r/((d_GR + d3D)·c)` with
  `d_GR = sqrt(d2D² + (h_t + h_r)²)`; AoD and AoA of the LOS path; EoD = EoA = `-atan2(h_t + h_r, d2D)`.
  Reflection coefficients per frequency of eqs. (7.6-38), (7.6-39) with the permittivity
  `eps = gr_eps_re - j·gr_eps_im` (the sign of eq. (7.6-40); `gr_eps_im` >= 0 as from [[lsp_generate]]), `cos
  theta = -(h_t + h_r)/d_GR'`, `sin² theta = (d2D/d_GR')²`, `d_GR' = max(d_GR, 1e-5)`; a coefficient with a zero
  denominator (only at grazing with `eps = 1`) is 0. Polarisation-averaged reflection power
  `R² = (abs(R_par)² + abs(R_perp)²)/2`
- Ground-reflection power: `"38.901"`, `"38.901_MF"`: `K/(K+1)·(d3D/d_GR')²·R²` (eq. (7.6-32)), the other slots
  unchanged, so the total power is above 1 (no renormalisation; median +1.6 dB in UMi LOS at 6 GHz); `"quadriga"`:
  `K/(K+1)·R²` (QuaDRiGa's two-ray channel, eq. (7.6-32) without the factor d3D/d_GR); its spread scaling splits the
  LOS power into `(1 - R²)·K/(K+1)` and `R²·K/(K+1)`, so the delay spread computed from the slot powers of a
  ground-reflection link is about 0.85 of the input. In NLOS scenarios the ground-reflection power is 0
- Coefficients from the slot powers, the same for all methods: LOS path amplitude `sqrt(power)` with the
  polarisation `diag(1, -1)` of eq. (7.5-29); ground reflection amplitude `sqrt(power)` with the polarisation
  `diag(R_par, -R_perp)/R`, `R = sqrt(R²)`, which applies the reflection exactly once and gives eq. (7.6-32);
  `R² = 0` gives a zero coefficient. A consumer without polarisation uses `sqrt(power)·exp(j·arg(R_par))`
  (`arg(0) = 0`) for the reflection: it keeps the reflection phase (about pi near grazing), without which the
  two-ray interference is reversed. NLOS clusters: amplitude `sqrt(power)`
- Permittivity: `gr_eps_re >= 1` and `gr_eps_im >= 0`; [[lsp_generate]] gives real parts of 3.0 to 44 between 0.5
  and 100 GHz; its fit falls below 1 above about 1.3 THz, where this function rejects it
- `NumClusters = 1` (`N = 0`): the LOS path only, with power 1 in all methods and all scenarios (K is not used),
  plus the ground reflection if enabled, with the power of the method's rule for `K/(K+1) = 1`; `rnd` is empty,
  `rnd_freq` and `lsp_anchor` are not needed, `n_cluster` entries are 0, `n_active` is 0
- Method `"38.901"`, per link with `N_l` active clusters, in degrees; `u_n` = block 0, `s_n` the sign block and
  `y_n` the offset block of the angle type, `z_n` the shadowing value, `r_tau = r_DS`, `zeta = LNS_ksi`,
  `K` = KF of `lsp` (LOS scenarios):
  `tau'_n = -r_tau·DS·ln(u_n)`, `tau_n = tau'_n - min(tau')`; the first cluster `f` has the smallest `tau'_n`
  (lowest index on ties). LOS: `K_lim = max(10·lg K, -5)`,
  `C_tau = 0.7705 - 0.0433·K_lim + 0.0002·K_lim² + 0.000017·K_lim³`, delay `tau_n / C_tau` (eq. (7.5-4)); NLOS: delay
  `tau_n`.
  `P_n = P'_n / sum(P')`, `P'_n = exp(-tau_n·(r_tau - 1)/(r_tau·DS))·10^(-zeta·z_n/10)` (eqs. (7.5-5), (7.5-6),
  computed in the log domain). Clusters with `P_n < 10^(-2.5)·max(P)` are removed: power 0, delay and angles kept,
  no renormalisation. Output power `P_n/(K+1)`.
  Angle mapping with `Pa_n = P_n/(K+1) + (n = f ? K/(K+1) : 0)` (eq. (7.5-8); NLOS `Pa_n = P_n`):
  azimuth `a_n = 2·(AS/1.4)·sqrt(-ln(Pa_n/max(Pa)))/C_phi`, zenith `a_n = -ZS·ln(Pa_n/max(Pa))/C_theta`, with
  `C_phi = C_phi^NLOS(N_l)·(1.1035 - 0.028·K_lim - 0.002·K_lim² + 0.0001·K_lim³)` and
  `C_theta = C_theta^NLOS(N_l)·(1.3086 + 0.0339·K_lim - 0.0077·K_lim² + 0.0002·K_lim³)` in LOS (NLOS: the table values);
  `C_phi^NLOS`, `C_theta^NLOS` from TR 38.901 Tables 7.5-2 and 7.5-4, linearly interpolated between the table
  entries and constant beyond both ends.
  `v_n = X_n·a_n + Y_n`, `X_n = +1` if `s_n >= 0`, else -1, `Y_n = y_n·AS/7`.
  NLOS: AOD = LOS AOD + v_n, AOA = LOS AOA + v_n, ZOD = LOS ZOD + mu_offset,ZOD + v_n with
  `mu_offset,ZOD = -eod_offset_deg(l, 0)`, ZOA = (90 if `o2i(l)` else LOS ZOA) + v_n. LOS: angle = LOS angle +
  v_n - v_f (eqs. (7.5-12), (7.5-17): the first cluster at delay 0 in the LOS direction; `o2i` and the ZOD
  offset have no effect). Zenith angles are wrapped to [0, 360] and mapped to `360 - theta` in [180, 360] (ZOA
  and ZOD), azimuths wrapped
- Deviations of `"38.901"` from the TR text: the K-factor in eqs. (7.5-3), (7.5-10), (7.5-15) is limited to
  >= -5 dB (below, C_theta turns negative near -10 dB); for links far below -5 dB the ASA is then about 0.86 of the
  input. The delays are not sorted: the TR's sort of eq. (7.5-2) only relabels the clusters, and keeping the
  cluster of each field column keeps the clusters stable under spatial consistency
- Method `"38.901_MF"` (TR 38.901 clause 7.6.5.1 as written), per link with `N_l` active clusters, in degrees;
  `r = 1.5`; anchor spreads `DS0`, `ASD0`, `ASA0`, `ZSD0`, `ZSA0` from `lsp_anchor`; DS, ASD, ASA, ZSD, ZSA and `K`
  from `lsp` at frequency k (`K = 0` in NLOS scenarios); `u_n` = block 0, `y_n` the offset block of the angle type
  (blocks 5 to 8: Y_n, Z_n of the TR and, through `V = Phi(y)`, V_n, W_n), `z_{n,k}` the shadowing at frequency k,
  `zeta = LNS_ksi`.
  Step 5', the same at all frequencies: `tau'_n = -r·DS0·ln(u_n)`; azimuths
  `phi'_n = arg(exp(-j·r·AS0·y_n))`, i.e. `-r·AS0·y_n` wrapped to [-180, 180); zeniths
  `theta'_n = arg(exp(-j·r·ZS0·sgn(y_n)·ln(erfc(abs(y_n)/sqrt(2)))/sqrt(2)))`, which equals the TR's
  `sgn(V_n - 0.5)·ln(1 - 2·abs(V_n - 0.5))` with full precision near V = 0 or 1. LOS: cluster 0 (the TR's cluster
  1) has `tau'_0 = 0` and all four nominal angles 0.
  Step 6' per frequency: `P'_n = exp(-tau'_n·g_DS - (phi'_AOD,n·g_ASD/sqrt(2))² - (phi'_AOA,n·g_ASA/sqrt(2))² -
  sqrt(2)·abs(theta'_ZOD,n)·g_ZSD - sqrt(2)·abs(theta'_ZOA,n)·g_ZSA)·10^(-zeta·z_{n,k}/10)` (eq. (7.6.30a)) with
  `g_DS = max(r·DS0 - DS, 0)/(DS·r·DS0)`, `g_ASD = sqrt(max((r·ASD0)² - ASD², 0))/(ASD·r·ASD0)` (ASA likewise),
  `g_ZSD = max(r·ZSD0 - ZSD, 0)/(ZSD·r·ZSD0)` (ZSA likewise); a g factor is 0 where the spread at k is at least r
  times the anchor spread. `P_n = P'_n/sum(P')` (log domain), output power `P_n/(K+1)`; with the LOS slot
  `K/(K+1)` this is eq. (7.6.30h).
  Step 7' per frequency: delay `sqrt(1 + K/2)·tau'_n` (LOS) or `tau'_n` (NLOS), plus `abs_toa` in NLOS scenarios;
  angle = LOS angle + `sqrt(1 + K)·angle'_n` (LOS) or LOS angle + `angle'_n` (NLOS) for AOD, AOA, ZOD, ZOA
  (eqs. (7.6.30i) to (7.6.30m) without the ray terms); azimuths wrapped, zenith angles folded as in `"38.901"`
- `"38.901_MF"` follows the TR text literally: the nominal angles are wrapped in step 5' and scaled by `sqrt(1 + K)`
  in step 7', so in LOS an angle that crosses +-180 deg in step 5' jumps by `sqrt(1 + K)·360` deg before the final
  wrap (about 0.05 to 0.07 jumps per metre along LOS tracks); in LOS the delays and angles depend on the frequency
  through K (they are equal at all frequencies if K is); the NLOS delays are not shifted to start at 0 (no cluster
  at delay 0 in NLOS); no sort, no -25 dB removal, no C factors, no ZOD offset (mean EoD about 2 to 3 deg off the
  clause 7.5 statistics in UMi and UMa NLOS), no O2I rule; the per-cluster shadowing is drawn per frequency (TR
  clause 7.6.5); the anchor LSPs (TR: e.g. at 2 GHz) are an input; the zenith fold of clause 7.5 step 7 is applied
  to the final zenith angles, for which clause 7.6.5.1 states no rule
- Method `"quadriga"` (the paper's section II with the changes below), per link with `N_l` active clusters, in rad;
  `u_n` = block 0, `y^x_n` the offset block of angle type x (blocks 5 to 8: AOD, AOA, EOD, EOA); DS, the four
  angular spreads `AS_x` and `K` from `lsp` at frequency k (`K = 0` in NLOS scenarios); `R²_k` the reflection power
  and `G = 1` with ground reflection, else 0. Delays and angles are the same at all frequencies.
  Initial values (paper eqs. (1), (4), (5)): `tau~_n = -ln(u_n)` (unitless), `a~^x_n = (Phi(y^x_n) - 0.5)·pi` in
  (-pi/2, pi/2), `Phi(y) = 0.5·erfc(-y/sqrt(2))`.
  Centres: the LOS azimuths; the elevations `(1 - w)·LOS elevation + w·E_GR` with `w` the mean of `R²_k` over the
  frequencies in LOS scenarios with ground reflection, else `w = 0`. LOS path and ground reflection enter the scaling
  as fixed offsets (azimuth 0, elevation their elevation minus the centre).
  Powers per frequency (paper eqs. (6) to (12)): `P_n` proportional to `exp(-tau~_n·g_DS - (a~^AOD_n)²·g_AOD -
  (a~^AOA_n)²·g_AOA - abs(a~^EOD_n)·g_EOD - abs(a~^EOA_n)·g_EOA)`, normalised to `1/(K+1)`; no per-cluster shadowing.
  One frequency: `g_DS = r_tau - 1`, `g_AOD = g_AOA = 0.56`, `g_EOD = g_EOA = 0.76`. Several frequencies:
  `g_DS = -1.5·ln(1.2·d - 0.15)`, `d = DS_k/(max DS + min DS)` limited to [0.15, 0.85] (`r_tau - 1` if DS does not
  depend on the frequency); azimuth `g = -2.2·ln(1.5·b - 0.35)`, elevation `g = -3.4·ln(1.2·b - 0.1)`,
  `b = max(0.75·AS_k/max AS, 0.25)`.
  Scaling weights: LOS `K/(K+1)·(1 - G·R²)`, ground reflection `K/(K+1)·G·R²`, clusters `P_n` (sum 1).
  Delay scale per frequency (paper eqs. (13), (14)): the factor `s_k` for which the weighted delays (LOS 0, ground
  reflection its delay, clusters `s_k·tau~_n`) have the delay spread DS_k, in closed form (1 if the clusters have no
  spread; with ground reflection at least the factor that minimises the spread); delays `s·tau~_n` in [s] with `s`
  the mean of `s_k`.
  Angle scale per type and frequency: the input spread over the wrapped spread of the initial angles, then up to 9
  updates of the factor in dB until the linear power-weighted spread of the scaled cluster offsets together with the
  fixed offsets equals the input to 1e-4 dB; the mean over the frequencies, at most 3 (azimuth) and 1.5 (elevation)
  (paper eq. (18)). Scaled elevation offsets are mirrored at +-pi/2.
  Angles: AOD, AOA = LOS azimuth + offset (wrapped); EOD = centre + `delta` + offset, EOA = centre + offset, both
  mirrored at the poles (the azimuth is kept); `delta` = the mean of `eod_offset_deg` over the frequencies in NLOS
  scenarios (TR eq. (7.5-19) with mu_offset,ZOD = -offset), 0 in LOS scenarios.
  Absolute TOA in NLOS scenarios: where `abs_toa(l) > 0` the NLOS delays are shifted so that the first is at
  `abs_toa(l)`
- Changes of `"quadriga"` against the paper and QuaDRiGa: the LOS direction is added to the scaled offsets (azimuth
  wrapped, elevation mirrored at the poles) instead of the 3-D rotation of the paper (which widens the elevation
  spread of departures); the EoD offset of the TR is added in NLOS scenarios; no weakening of the arrival elevations
  for links with a large height difference; no pull of the NLOS elevations towards the ground reflection in NLOS
  scenarios; no fixed angles for one or two NLOS clusters. `o2i` is not used
- Properties of `"quadriga"`: at one frequency the delay spread of the output (all slots) equals DS, also with the
  absolute TOA; with ground reflection the scaling uses the split weights above, so the delay spread from the slot
  powers is smaller (about 0.85 of DS in UMi and UMa LOS at 6 GHz, medium ground), and the elevation spreads cannot
  fall below the spread of the LOS path and the reflection alone (the target is then not reached). Below the caps
  the linear power-weighted spread of the angle offsets equals the input; with the circular definition of
  [[calc_angular_spreads_sphere]] the medians are 1.00 for ASD, ESD, ESA and ASA in NLOS, while in LOS the LOS power
  bounds the ASA (median 0.86 UMi, 0.59 UMa). With several frequencies the averaged scales leave a per-frequency
  spread error of about +-15 to 20 %. Continuous along tracks; delays and angles independent of the frequency; the
  relation between angle offset and power is weaker than in clause 7.5
- Swap reciprocity: with mirrored random values (dual-mobility values of [[ssf_draw]], or the blocks 1 and 2, 3 and
  4, 5 and 6, 7 and 8 exchanged) and mirrored LSPs (ASD and ASA, ESD and ESA exchanged, also in `lsp_anchor`),
  swapping `tx_pos` and `rx_pos` gives the same delays and powers with departure and arrival angles exchanged;
  `"38.901"` needs `eod_offset_deg` = 0 and no O2I link, `"quadriga"` `eod_offset_deg` = 0
- Properties of the methods (typical values for UMi and UMa at 6 and 30 GHz):

| Property | `38.901` | `38.901_MF` | `quadriga` |
| --- | --- | --- | --- |
| Output/input spread, median | DS 0.96-1.02, ASD 0.95-1.02, ASA 0.99 (NLOS) / 0.80-0.89 (LOS), ESD 0.97-1.04, ESA 0.97-1.03 | NLOS 0.88-0.94, LOS 0.44-0.77 | DS 1.00, ASD 1.00, ESD 1.00, ESA 1.00, ASA 1.00 (NLOS) / 0.59-0.86 (LOS) |
| Continuous along a track | no: in LOS the first cluster changes (0.03-0.11 jumps/m), the -25 dB rule switches clusters (0.03-0.19 /m), signs and offsets must be frozen per segment; nearby UTs see sign flips (0.5-1.5 jumps per m of separation) | NLOS yes; LOS no (0.05-0.07 /m, wrap before the scaling by sqrt(1+K)) | yes |
| Delays and angles equal at all frequencies | one frequency only | NLOS yes; LOS only if K does not depend on frequency | yes |
| Exact spreads | no (C factors calibrated) | no | DS exact at one frequency; AS exact below the caps (linear definition); averaged scales with several frequencies |
| Swap-reciprocal (mirrored fields and LSPs) | yes, if `eod_offset_deg` = 0 and no O2I link | yes | yes, if `eod_offset_deg` = 0 |
| Per-cluster shadowing | yes (zeta) | yes (Q_n) | no |
| TR terms mu_offset,ZOD (NLOS), O2I ZOA 90 deg | both | neither | offset only |
| `NumClusters = 1` | LOS path only (+ GR, eq. (7.6-32)) | LOS path only (+ GR, eq. (7.6-32)) | LOS path only (+ GR, R²) |
| Ground reflection (coefficient powers) | TR 7.6.8: K/(K+1)·(d3D/d_GR)²·R² | TR 7.6.8 | K/(K+1)·R²; power split inside the scaling |

- Recommended use: `"38.901"` for drop-based calibration at one frequency; `"quadriga"` for tracks, several
  frequencies or device-to-device links; `"38.901_MF"` where the multi-frequency method of the TR is required
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario ([[lsp_generate]], [[ssf_draw]]); this is
  not checked beyond the sizes. `scenpar` is completed with the defaults of missing keys and validated by
  [[scenpar_validate]]; its messages pass through unchanged
- `n_link = 0` gives outputs with 0 rows. Outputs are resized only if their size differs, and every element is
  written. Outputs must not share memory with the inputs or with each other
- Validation order: `method`, positions, `scenpar`, `lsp`, one frequency for `"38.901"`, `rnd`, `rnd_freq`,
  `lsp_anchor`, `eod_offset_deg`, `abs_toa`, ground-reflection inputs and heights, `o2i`, `n_cluster`; all checks
  run before any output is written

## Declaration:
```
void quadriga_lib::ssf_clusters(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::cube &lsp,
    const arma::mat &rnd,
    arma::cube &delay,
    arma::cube &power,
    arma::cube &aod,
    arma::cube &eod,
    arma::cube &aoa,
    arma::cube &eoa,
    const std::string &method = "38.901",
    const arma::cube *rnd_freq = nullptr,
    const arma::cube *lsp_anchor = nullptr,
    const arma::mat *eod_offset_deg = nullptr,
    const arma::vec *abs_toa = nullptr,
    const arma::mat *gr_eps_re = nullptr,
    const arma::mat *gr_eps_im = nullptr,
    const std::vector<bool> *o2i = nullptr,
    const arma::uvec *n_cluster = nullptr,
    arma::uvec *n_active = nullptr);
```

## Inputs:
- **`scenpar`** — Scenario parameters as returned by [[conf_file_read]] or [[scenpar_validate]]; used: `NumClusters`,
  `NumClusters_min`, `KF_mu` (LOS condition), `r_DS`, `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`lsp`** — Large-scale parameters from [[lsp_generate]], finite; DS, ASD, ASA, ESD, ESA (slices 0, 3, 4, 5, 6)
  > 0, KF (slice 1) >= 0; slices 2 and 7 are not used; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set from [[ssf_draw]] (`set_id = -1`), finite, columns 0 to N-1 in [0, 1];
  `[n_link, 9·N]` (any empty matrix for `N = 0`)
- **`method`** *(optional)* — `"38.901"` (default), `"38.901_MF"` or `"quadriga"`
- **`rnd_freq`** *(optional)* — Per-frequency sets from [[ssf_draw]] (`set_id = k` in slice `k`), finite;
  required by `"38.901"` and `"38.901_MF"` if N > 0; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency from [[lsp_generate]], finite, spreads > 0; required
  by `"38.901_MF"` if N > 0; `[n_link, 1, 8]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg] from [[lsp_generate]], positive = up, finite;
  omitted = 0; `[n_link, n_freq]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s] from [[lsp_generate]], finite and >= 0; omitted = 0;
  `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Relative permittivity of the ground, real part >= 1 and imaginary
  part >= 0, from [[lsp_generate]]; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** *(optional)* — O2I flag per link (mean ZOA 90 deg in `"38.901"`); omitted = no O2I link; `[n_link]`
- **`n_cluster`** *(optional)* — Number of active NLOS clusters per link in [N_lo, N] with
  `N_lo = min(max(1, NumClusters_min - 1), N)` (TR 38.901 Table 7.6.15-1 where the file states the range);
  omitted = N; `[n_link]`

## Outputs:
- **`delay`** — Delays in [s] relative to the LOS delay; `[n_link, n_slot, n_fd]`, `n_fd = n_freq` for
  `"38.901_MF"`, else 1
- **`power`** — Coefficient powers (linear) per frequency; `[n_link, n_slot, n_freq]`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in [rad], in [-pi, pi); `[n_link, n_slot, n_fd]`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in [rad], in [-pi/2, pi/2]; `[n_link, n_slot, n_fd]`
- **`n_active`** *(optional)* — Number of NLOS clusters with power > 0; `[n_link]`

## See also:
- [[ssf_draw]] (the random values `rnd` and `rnd_freq`)
- [[ssf_init_phases]] (the phases of the random generators)
- [[lsp_generate]] (the LSPs, the EoD offset, the absolute TOA and the ground permittivity)
- [[lsf_draw]] (the random values of the LSPs)
- [[calc_delay_spread]], [[calc_angular_spreads_sphere]] (spreads of the output)
MD!*/

void quadriga_lib::ssf_clusters(const std::unordered_map<std::string, double> &scenpar,
                                const arma::mat &tx_pos,
                                const arma::mat &rx_pos,
                                const arma::cube &lsp,
                                const arma::mat &rnd,
                                arma::cube &delay,
                                arma::cube &power,
                                arma::cube &aod,
                                arma::cube &eod,
                                arma::cube &aoa,
                                arma::cube &eoa,
                                const std::string &method,
                                const arma::cube *rnd_freq,
                                const arma::cube *lsp_anchor,
                                const arma::mat *eod_offset_deg,
                                const arma::vec *abs_toa,
                                const arma::mat *gr_eps_re,
                                const arma::mat *gr_eps_im,
                                const std::vector<bool> *o2i,
                                const arma::uvec *n_cluster,
                                arma::uvec *n_active)
{
    // Validation, before any output is touched
    const bool m_tr = method == "38.901", m_mf = method == "38.901_MF", m_qd = method == "quadriga";
    if (!m_tr && !m_mf && !m_qd)
        throw std::invalid_argument("Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.");
    qd_gbsm_check_links(&tx_pos, &rx_pos);
    const arma::uword n_link = rx_pos.n_rows;

    std::unordered_map<std::string, double> sp = scenpar;
    quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, true);
    const arma::uword N = (arma::uword)sp.at("NumClusters") - 1;
    const bool G = sp.at("GR_enabled") > 0.0;

    if (lsp.n_rows != n_link || lsp.n_cols < 1 || lsp.n_slices != 8)
        throw std::invalid_argument("Input 'lsp' must have size [n_link, n_freq, 8].");
    if (!lsp.is_finite())
        throw std::invalid_argument("Input 'lsp' must be finite.");
    const arma::uword F = lsp.n_cols;
    auto spreads_positive = [](const arma::cube &c, const char *msg)
    {
        for (arma::uword s : {0u, 3u, 4u, 5u, 6u})
            if (c.n_elem != 0 && !arma::all(arma::vectorise(c.slice(s)) > 0.0))
                throw std::invalid_argument(msg);
    };
    spreads_positive(lsp, "Delay and angular spreads in 'lsp' must be > 0.");
    if (lsp.n_elem != 0 && arma::any(arma::vectorise(lsp.slice(1)) < 0.0))
        throw std::invalid_argument("K-factor in 'lsp' must be >= 0.");
    if (m_tr && F > 1)
        throw std::invalid_argument(
            "Method '38.901' supports one frequency only; use '38.901_MF' or 'quadriga' for several frequencies.");

    if (N == 0 ? rnd.n_elem != 0 : (rnd.n_rows != n_link || rnd.n_cols != 9 * N))
        throw std::invalid_argument("Input 'rnd' must have size [n_link, " + std::to_string(9 * N) + "].");
    if (!rnd.is_finite())
        throw std::invalid_argument("Input 'rnd' must be finite.");
    for (arma::uword i = 0; i < n_link * N; ++i)
        if (!(rnd[i] >= 0.0 && rnd[i] <= 1.0))
            throw std::invalid_argument("Columns 0 to N-1 of 'rnd' (cluster delays) must be in [0, 1].");

    if ((m_tr || m_mf) && N > 0)
    {
        if (rnd_freq == nullptr)
            throw std::invalid_argument("Input 'rnd_freq' is required for method '" + method + "'.");
        if (rnd_freq->n_rows != n_link || rnd_freq->n_cols != N || rnd_freq->n_slices != F)
            throw std::invalid_argument("Input 'rnd_freq' must have size [n_link, " + std::to_string(N) +
                                        ", n_freq].");
        if (!rnd_freq->is_finite())
            throw std::invalid_argument("Input 'rnd_freq' must be finite.");
    }
    if (m_mf && N > 0)
    {
        if (lsp_anchor == nullptr)
            throw std::invalid_argument("Input 'lsp_anchor' is required for method '38.901_MF'.");
        if (lsp_anchor->n_rows != n_link || lsp_anchor->n_cols != 1 || lsp_anchor->n_slices != 8)
            throw std::invalid_argument("Input 'lsp_anchor' must have size [n_link, 1, 8].");
        if (!lsp_anchor->is_finite())
            throw std::invalid_argument("Input 'lsp_anchor' must be finite.");
        spreads_positive(*lsp_anchor, "Delay and angular spreads in 'lsp_anchor' must be > 0.");
    }
    if ((m_tr || m_qd) && eod_offset_deg != nullptr)
    {
        if (eod_offset_deg->n_rows != n_link || eod_offset_deg->n_cols != F)
            throw std::invalid_argument("Input 'eod_offset_deg' must have size [n_link, n_freq].");
        if (!eod_offset_deg->is_finite())
            throw std::invalid_argument("Input 'eod_offset_deg' must be finite.");
    }
    if (abs_toa != nullptr)
    {
        if (abs_toa->n_elem != n_link)
            throw std::invalid_argument("Number of elements in 'abs_toa' does not match number of rows in 'rx_pos'.");
        if (!abs_toa->is_finite() || arma::any(*abs_toa < 0.0))
            throw std::invalid_argument("Input 'abs_toa' must be finite and >= 0.");
    }
    if (G)
    {
        if (gr_eps_re == nullptr || gr_eps_im == nullptr)
            throw std::invalid_argument("Inputs 'gr_eps_re' and 'gr_eps_im' are required when 'GR_enabled' is set.");
        if (gr_eps_re->n_rows != n_link || gr_eps_re->n_cols != F || gr_eps_im->n_rows != n_link ||
            gr_eps_im->n_cols != F)
            throw std::invalid_argument("Inputs 'gr_eps_re' and 'gr_eps_im' must have size [n_link, n_freq].");
        if (!gr_eps_re->is_finite() || !gr_eps_im->is_finite())
            throw std::invalid_argument("Inputs 'gr_eps_re' and 'gr_eps_im' must be finite.");
        if (arma::any(arma::vectorise(*gr_eps_re) < 1.0) || arma::any(arma::vectorise(*gr_eps_im) < 0.0))
            throw std::invalid_argument("Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.");
        if (arma::any(rx_pos.col(2) < 0.0) || arma::any(tx_pos.col(2) < 0.0))
            throw std::invalid_argument("Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'.");
    }
    if (m_tr && o2i != nullptr && o2i->size() != n_link)
        throw std::invalid_argument("Number of elements in 'o2i' does not match number of rows in 'rx_pos'.");
    if (n_cluster != nullptr)
    {
        if (n_cluster->n_elem != n_link)
            throw std::invalid_argument("Number of elements in 'n_cluster' does not match number of rows in 'rx_pos'.");
        const double n_min = sp.at("NumClusters_min");
        const arma::uword N_lo = std::min(n_min > 2.0 ? (arma::uword)n_min - 1 : (arma::uword)1, N);
        if (arma::any(*n_cluster < N_lo) || arma::any(*n_cluster > N))
            throw std::invalid_argument("Entries of 'n_cluster' must be in [" + std::to_string(N_lo) + ", " +
                                        std::to_string(N) + "].");
    }

    // Outputs
    const arma::uword n_slot = 1 + (G ? 1 : 0) + N, n_fd = m_mf ? F : 1;
    for (arma::cube *c : {&delay, &aod, &eod, &aoa, &eoa})
        if (c->n_rows != n_link || c->n_cols != n_slot || c->n_slices != n_fd)
            c->set_size(n_link, n_slot, n_fd);
    if (power.n_rows != n_link || power.n_cols != n_slot || power.n_slices != F)
        power.set_size(n_link, n_slot, F);
    if (n_active != nullptr && n_active->n_elem != n_link)
        n_active->set_size(n_link);

    if (n_link == 0)
        return;

    // Per link
    const bool los = sp.at("KF_mu") > -30.0;
    const double r_tau = sp.at("r_DS"), zeta = sp.at("LNS_ksi"), r2d = 180.0 / qd_pi;
    const bool tx_single = tx_pos.n_rows != n_link;
    arma::cube *ang_out[4] = {&aod, &aoa, &eod, &eoa};

#pragma omp parallel for
    for (long long i = 0; i < (long long)n_link; ++i)
    {
        const arma::uword l = (arma::uword)i, t = tx_single ? 0 : l;
        const arma::uword N_l = n_cluster != nullptr ? (*n_cluster)(l) : N;

        // LOS geometry; the angles of slot 0 and of the inactive slots
        const double vx = rx_pos(l, 0) - tx_pos(t, 0), vy = rx_pos(l, 1) - tx_pos(t, 1);
        const double vz = rx_pos(l, 2) - tx_pos(t, 2);
        const double d2D = std::sqrt(vx * vx + vy * vy), d3D = std::sqrt(vx * vx + vy * vy + vz * vz);
        const double aod_l = std::atan2(vy, vx), eod_l = std::atan2(vz, d2D);
        const double los_ang[4] = {aod_l, qd_wrap(aod_l + qd_pi), eod_l, -eod_l}; // AOD, AOA, EOD, EOA

        // Ground reflection: delay, elevation, the power factor (TR methods (d3D/d_GR)², eq. (7.6-32); "quadriga" 1)
        // and the polarisation-averaged reflection power per frequency
        double tau_gr = 0.0, e_gr = 0.0, g_gr = 0.0;
        std::vector<double> R2(G ? F : 0);
        if (G)
        {
            const double ht = tx_pos(t, 2), hr = rx_pos(l, 2);
            const double d_gr = std::sqrt(d2D * d2D + (ht + hr) * (ht + hr)), d_gr1 = std::max(d_gr, 1e-5);
            tau_gr = (d_gr + d3D > 0.0) ? 4.0 * ht * hr / ((d_gr + d3D) * 299792458.0) : 0.0;
            e_gr = -std::atan2(ht + hr, d2D);
            g_gr = m_qd ? 1.0 : (d3D / d_gr1) * (d3D / d_gr1);
            const double ct = -(ht + hr) / d_gr1, s2 = (d2D / d_gr1) * (d2D / d_gr1);
            for (arma::uword k = 0; k < F; ++k)
            {
                const std::complex<double> eps((*gr_eps_re)(l, k), -(*gr_eps_im)(l, k));
                const std::complex<double> q = std::sqrt(eps - s2), a = eps * ct, d_par = a - q, d_per = ct - q;
                const std::complex<double> R_par = (d_par == 0.0) ? 0.0 : (a + q) / d_par;
                const std::complex<double> R_per = (d_per == 0.0) ? 0.0 : (ct + q) / d_per;
                R2[k] = 0.5 * (std::norm(R_par) + std::norm(R_per));
            }
        }

        // NLOS clusters by the method
        std::vector<double> c_delay(N_l * n_fd), c_power(N_l * F), c_ang(4 * N_l * n_fd);
        if (N_l > 0)
        {
            std::vector<double> x(9 * N), z(m_qd ? 0 : N * F);
            for (arma::uword c = 0; c < 9 * N; ++c)
                x[c] = rnd(l, c);
            for (arma::uword k = 0; k < F && !m_qd; ++k)
                for (arma::uword n = 0; n < N; ++n)
                    z[k * N + n] = (*rnd_freq)(l, n, k);
            double centre[4] = {los_ang[0] * r2d, los_ang[1] * r2d, 90.0 - eod_l * r2d, 90.0 + eod_l * r2d};
            if (m_tr)
            {
                const double as[4] = {lsp(l, 0, 3), lsp(l, 0, 4), lsp(l, 0, 5), lsp(l, 0, 6)};
                if (!los)
                {
                    centre[2] -= (eod_offset_deg != nullptr) ? (*eod_offset_deg)(l, 0) : 0.0; // mu_offset,ZOD
                    centre[3] = (o2i != nullptr && (*o2i)[l]) ? 90.0 : centre[3];
                }
                qd_ssf_tr(x.data(), z.data(), N, N_l, los, r_tau, zeta, lsp(l, 0, 0), los ? lsp(l, 0, 1) : 0.0, as,
                          centre, c_delay.data(), c_power.data(), c_ang.data());
            }
            else if (m_mf)
            {
                const arma::uword si[5] = {0, 3, 4, 5, 6}; // DS, ASD, ASA, ZSD, ZSA
                double sp0[5];
                std::vector<double> spk(5 * F), kf(F);
                for (arma::uword i = 0; i < 5; ++i)
                {
                    sp0[i] = (*lsp_anchor)(l, 0, si[i]);
                    for (arma::uword k = 0; k < F; ++k)
                        spk[5 * k + i] = lsp(l, k, si[i]);
                }
                for (arma::uword k = 0; k < F; ++k)
                    kf[k] = los ? lsp(l, k, 1) : 0.0;
                qd_ssf_mf(x.data(), z.data(), N, N_l, F, los, zeta, sp0, spk.data(), kf.data(), centre,
                          c_delay.data(), c_power.data(), c_ang.data());
            }
            else
            {
                // Spreads in [s] and [rad]; EoD offset averaged over the frequencies (NLOS scenarios)
                std::vector<double> spk(5 * F), kf(F);
                double delta = 0.0;
                for (arma::uword k = 0; k < F; ++k)
                {
                    spk[5 * k] = lsp(l, k, 0);
                    for (arma::uword i = 1; i < 5; ++i)
                        spk[5 * k + i] = lsp(l, k, 2 + i) / r2d;
                    kf[k] = los ? lsp(l, k, 1) : 0.0;
                    delta += (!los && eod_offset_deg != nullptr) ? (*eod_offset_deg)(l, k) / r2d / (double)F : 0.0;
                }
                const double toa = (!los && abs_toa != nullptr) ? (*abs_toa)(l) : 0.0;
                qd_ssf_quadriga(x.data(), N, N_l, F, los, r_tau, spk.data(), kf.data(), los_ang, G, tau_gr, e_gr,
                                R2.data(), delta, toa, c_delay.data(), c_power.data(), c_ang.data());
            }

            // Absolute TOA in NLOS scenarios, TR methods (TR 38.901 eq. (7.6-43)); "quadriga" shifts in its kernel
            if (!los && abs_toa != nullptr && !m_qd)
                for (double &d : c_delay)
                    d += (*abs_toa)(l);
        }

        // Write the slots
        arma::uword act = 0;
        for (arma::uword s = 0; s < n_slot; ++s)
        {
            const bool is_gr = G && s == 1, is_cl = s > (G ? 1 : 0);
            const arma::uword n = is_cl ? s - 1 - (G ? 1 : 0) : 0;
            const bool active = is_cl && n < N_l;
            for (arma::uword k = 0; k < n_fd; ++k)
            {
                delay(l, s, k) = active ? c_delay[k * N_l + n] : (is_gr ? tau_gr : 0.0);
                for (arma::uword a = 0; a < 4; ++a)
                    (*ang_out[a])(l, s, k) = active ? c_ang[(k * 4 + a) * N_l + n]
                                                    : ((is_gr && a > 1) ? e_gr : los_ang[a]);
            }
            bool on = false;
            for (arma::uword k = 0; k < F; ++k)
            {
                // Coefficient power of the LOS path: K/(K+1) in LOS scenarios, 0 in NLOS, 1 without NLOS clusters
                const double p_los = (N == 0) ? 1.0 : (los ? lsp(l, k, 1) / (lsp(l, k, 1) + 1.0) : 0.0);
                const double p = active ? c_power[k * N_l + n] : (is_cl ? 0.0 : (is_gr ? p_los * g_gr * R2[k] : p_los));
                power(l, s, k) = p;
                on = on || p > 0.0;
            }
            act += (active && on) ? 1 : 0;
        }
        if (n_active != nullptr)
            (*n_active)(l) = act;
    }
}
