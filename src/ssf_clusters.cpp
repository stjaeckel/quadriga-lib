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
    // Azimuth wrap to [-pi, pi)
    inline double qd_wrap(double x)
    {
        const double pi = arma::datum::pi;
        return x - 2.0 * pi * std::floor((x + pi) / (2.0 * pi));
    }

    // Zenith angle in [deg] to elevation in [rad]: wrapped to [0, 360), [180, 360) -> 360 - t (TR 38.901 clause 7.5
    // step 7), elevation = 90 - zenith
    inline double qd_elev(double t)
    {
        t -= 360.0 * std::floor(t / 360.0);
        return (90.0 - (t > 180.0 ? 360.0 - t : t)) * arma::datum::pi / 180.0;
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
        std::vector<double> tau(N_l), lam(N_l), P(N_l), Pa(N_l), rho(N_l), v(N_l);
        arma::uword f = 0;
        for (arma::uword n = 0; n < N_l; ++n)
        {
            tau[n] = -r_tau * ds * std::log(std::max(x[n], DBL_MIN));
            f = (tau[n] < tau[f]) ? n : f;
        }
        const double t_min = tau[f];
        // K limited to >= -5 dB in the polynomials of eqs. (7.5-3), (7.5-10), (7.5-15): C_theta falls to 0 near
        // -10 dB and turns negative below; for links far below -5 dB the ASA is then about 0.86 of the input
        const double k_db = std::max(10.0 * std::log10(kf), -5.0);
        const double k2 = k_db * k_db, k3 = k2 * k_db;
        const double c_tau = 0.7705 - 0.0433 * k_db + 0.0002 * k2 + 0.000017 * k3;

        // Step 6: powers, eqs. (7.5-5) to (7.5-8), normalised in the log domain
        const double p_th = 0.0031622776601683794; // 10^(-2.5): -25 dB removal threshold
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
        // -25 dB removal without renormalisation; removed clusters keep delay and angles and stay in max(Pa) and rho
        for (arma::uword n = 0; n < N_l; ++n)
        {
            Pa[n] = los ? P[n] * k1 + (n == f ? kk : 0.0) : P[n];
            pa_max = std::max(pa_max, Pa[n]);
            power[n] = (P[n] < p_th * p_max) ? 0.0 : P[n] * k1;
        }
        for (arma::uword n = 0; n < N_l; ++n) // ln(Pa_n / max(Pa)) of eqs. (7.5-9), (7.5-14), one value per cluster
            rho[n] = std::log(std::max(Pa[n], DBL_MIN) / pa_max);

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
                const double a1 = (a < 2) ? 2.0 * (as[a] / 1.4) * std::sqrt(-rho[n]) / c_phi : -as[a] * rho[n] / c_theta;
                v[n] = (x[(1 + a) * N + n] >= 0.0 ? a1 : -a1) + x[(5 + a) * N + n] * as[a] / 7.0;
            }
            const double v_f = v[f];
            for (arma::uword n = 0; n < N_l; ++n)
            {
                const double t = centre[a] + (los ? v[n] - v_f : v[n]); // eqs. (7.5-12), (7.5-17) align cluster f
                ang[a * N_l + n] = (a < 2) ? qd_wrap(t * arma::datum::pi / 180.0) : qd_elev(t);
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
    // The clause is followed as written: no minimum subtraction of the delays (no NLOS cluster at delay 0), no sort,
    // no -25 dB removal, no C factors, no ZOD offset, no O2I rule; the shadowing Q_n is drawn per frequency
    // (clause 7.6.5). The nominal angles are wrapped in step 5' and scaled by sqrt(1 + K) in step 7', so in LOS an
    // angle that crosses +-180 deg in step 5' jumps by sqrt(1 + K)·360 deg before the final wrap, and the LOS delays
    // and angles depend on the frequency through K. Clause 7.6.5.1 states no zenith fold; the fold of clause 7.5
    // step 7 is applied to the final zenith angles.
    void qd_ssf_mf(const double *x, const double *z, arma::uword N, arma::uword N_l, arma::uword F, bool los,
                   double zeta, const double *sp0, const double *sp, const double *kf, const double *centre,
                   double *delay, double *power, double *ang)
    {
        const double r = 1.5, s2 = std::sqrt(2.0);
        auto wrap180 = [](double v) // principal value in degrees, the domain of step 5' (qd_wrap works in rad)
        { return v - 360.0 * std::floor((v + 180.0) / 360.0); };

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
                    ang[(k * 4 + a) * N_l + n] = (a < 2) ? qd_wrap(t * arma::datum::pi / 180.0) : qd_elev(t);
                }
            }
        }
    }

    // Clusters of one link by the QuaDRiGa method (paper: Jaeckel et al., 2019, section II), all frequencies
    // (method "quadriga")
    // - x: values of the shared set of the link, column b·N + n; N_l active clusters; F frequencies
    // - sp[k·5 + i] = {DS [s], ASD, ASA, ESD, ESA [rad]} at frequency k; kf[k] linear K (0 in NLOS)
    // - los_ang = {AOD, AOA, EOD, EOA} [rad]; gr: ground reflection with delay tau_gr [s], elevation e_gr [rad] and
    //   reflection power r2[k]; delta: EoD offset [rad]
    // - Outputs for n < N_l: delay[n] [s] (relative, without abs TOA), power[k·N_l + n] coefficient power,
    //   ang[a·N_l + n] [rad] for a = 0 AOD, 1 AOA, 2 EOD, 3 EOA
    // Algorithm: initial delays -ln(u) and angles (Phi(y) - 0.5)·pi; powers from the product of paper eq. (6) with
    // the coefficients of eqs. (7) to (11); one delay scale per frequency such that the power-weighted delays (LOS 0,
    // ground reflection, clusters) have the spread DS (eqs. (13), (14), closed form); per angle type an angle scale
    // iterated in dB on the linear power-weighted spread with the LOS path and the ground reflection as fixed offsets;
    // the scales averaged over the frequencies (angle scales capped at 3 and 1.5, eq. (18)), so the delays and angles
    // are the same at all frequencies and the per-frequency spreads are matched on average only.
    // Changes against the paper: the LOS direction is added to the scaled offsets (azimuth wrapped, elevation
    // mirrored at the poles) instead of the 3-D rotation, which widens the departure elevation spread; the EoD offset
    // of TR eq. (7.5-19) on the NLOS departures of NLOS scenarios; no weakening of the arrival elevations for large
    // height differences; the elevation centre is not pulled towards the ground reflection in NLOS scenarios (it has
    // zero power there); no fixed angles for one or two NLOS clusters.
    // Properties: without ground reflection the delay spread of the output equals DS at one frequency; with ground
    // reflection the scaling uses the split K/(K+1)·(1 - R², R²) of the LOS power while the output carries the
    // coefficient powers, so the delay spread from the slot powers is smaller (about 0.85 of DS for typical LOS
    // links). Below the caps the linear power-weighted angle spread equals the input; elevation spreads below the
    // spread of the LOS path and the reflection alone cannot be reached.
    void qd_ssf_quadriga(const double *x, arma::uword N, arma::uword N_l, arma::uword F, bool los, double r_tau,
                         const double *sp, const double *kf, const double *los_ang, bool gr, double tau_gr,
                         double e_gr, const double *r2, double delta, double *delay, double *power, double *ang)
    {
        const double pi = arma::datum::pi;
        auto efold = [pi](double e) // elevation mirrored at the poles
        {
            e = qd_wrap(e);
            return e > 0.5 * pi ? pi - e : (e < -0.5 * pi ? -pi - e : e);
        };

        // Initial delays and angles, paper eqs. (1), (4), (5); (Phi(y) - 0.5)·pi = erf(y/sqrt(2))·pi/2
        std::vector<double> tt(N_l), at(4 * N_l), lam(N_l), P(N_l), A(N_l + 2), wt(N_l + 2);
        for (arma::uword n = 0; n < N_l; ++n)
        {
            tt[n] = -std::log(std::max(x[n], DBL_MIN));
            for (arma::uword a = 0; a < 4; ++a)
                at[a * N_l + n] = 0.5 * pi * std::erf(x[(5 + a) * N + n] / std::sqrt(2.0));
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

            // Delay scale, paper eqs. (13), (14), in closed form with the ground reflection as a fixed path; clusters
            // without delay spread (one active cluster or equal values) have nothing to scale: scale 0, delay 0
            const double den = c - e * e, a2 = s[0] * s[0], B = s_gr * tau_gr * tau_gr, D = s_gr * tau_gr;
            const double t = std::max(a2 * c - a2 * e * e - B * c + B * e * e + c * D * D, 0.0);
            s_tau += ((den > 0.0) ? (gr ? (std::sqrt(t) + D * e) / den : s[0] / std::sqrt(den)) : 0.0) / (double)F;

            // Angle scales: the input spread over the wrapped spread of the initial angles, then iterated on the
            // linear spread of the scaled cluster offsets with the fixed offsets of LOS and ground reflection
            for (arma::uword a = 0; a < 4; ++a)
            {
                wt[0] = s_los, A[0] = (a < 2) ? 0.0 : los_ang[a] - cen[a];
                if (gr)
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

        for (arma::uword n = 0; n < N_l; ++n)
            delay[n] = s_tau * tt[n];
    }
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

- Turns the LSPs of [[lsp_generate]] and the random values of [[ssf_draw]] into the clusters of each link, by one of
  three methods:
  - `"38.901"` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters
    25 dB below the strongest get power 0 (no renormalisation); the K-factor in the scaling factors is limited to
    >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` at the drop or segment
    start and keep them along the track
  - `"38.901_MF"`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `"quadriga"`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links

| Property | `"38.901"` | `"38.901_MF"` | `"quadriga"` |
| --- | --- | --- | --- |
| Continuous along tracks | no | NLOS only | yes |
| Same delays and angles at all frequencies | n/a | NLOS, or K the same at all frequencies | yes |
| Output spreads equal the input spreads | no | no | DS, and AS below the caps, at one frequency without ground reflection |
| Per-cluster shadowing | yes | yes, per frequency | no |

- Slots per link: 0 = LOS path, 1 = ground reflection if `GR_enabled > 0` (`G = 1`), then the `N = NumClusters - 1`
  NLOS clusters; `n_slot = NumClusters + G`. Slot `1 + G + n` belongs to column `n` of each block of `rnd` (no
  sorting). Clusters `n >= n_cluster(l)` (TR clause 7.6.15) get delay 0, power 0 and the LOS angles. Identify the
  LOS path by its slot, never by delay or path length: NLOS clusters of the TR methods can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever
  `lsp` holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; angles in [rad], GCS, elevation convention; slot 0 holds the
  geometric LOS direction. Powers are the coefficient powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters
  `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `"38.901"`, `"38.901_MF"` add `abs_toa(l)` to the NLOS
  delays; `"quadriga"` places the first NLOS cluster at `abs_toa(l)` (where `absTOA_mu <= -30`: only where
  `abs_toa(l) > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0. Slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`"quadriga"`) with `R² = (abs(R_par)² + abs(R_perp)²)/2` of TR eqs. (7.6-38),
  (7.6-39). The coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarisation
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios (`K/(K+1) = 1`, also in the ground-reflection
  power), plus the ground reflection if enabled; `rnd`, `rnd_freq` and `lsp_anchor` may be empty
- Swap reciprocity: with mirrored random values (dual mobility in [[ssf_draw]]) and mirrored LSPs, swapping `tx_pos`
  and `rx_pos` exchanges the departure and arrival angles (`"38.901"`: `eod_offset_deg` = 0 and no O2I link;
  `"quadriga"`: `eod_offset_deg` = 0)
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario (only sizes are checked); `scenpar` is
  validated by [[scenpar_validate]]. Outputs must not share memory with the inputs

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
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`, `KF_mu`, `absTOA_mu`, `r_DS`,
  `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in [m], one row for all links; z >= 0 with ground reflection; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[n_link, 3]`
- **`lsp`** — LSPs from [[lsp_generate]]; DS, ASD, ASA, ESD, ESA (slices 0, 3 to 6) > 0, KF (slice 1) >= 0;
  `"38.901"`: one frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [[ssf_draw]] (`set_id = -1`), columns 0 to N-1 in [0, 1]; `[n_link, 9·N]`
- **`method`** *(optional)* — `"38.901"` (default), `"38.901_MF"` or `"quadriga"`
- **`rnd_freq`** *(optional)* — Per-frequency sets of [[ssf_draw]], slice `k` from `set_id = k`; required by
  `"38.901"` and `"38.901_MF"`; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz) from [[lsp_generate]]; required by
  `"38.901_MF"`; `[n_link, 1, 8]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg] from [[lsp_generate]]; used by `"38.901"` and
  `"quadriga"` in NLOS scenarios; omitted = 0; `[n_link, n_freq]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s] from [[lsp_generate]], >= 0; required by `"quadriga"` in
  NLOS scenarios with `absTOA_mu > -30`; omitted = no offset; `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im` from
  [[lsp_generate]], real part >= 1, imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** *(optional)* — O2I flag per link (mean ZOA 90 deg); used by `"38.901"` in NLOS scenarios; omitted =
  none; `[n_link]`
- **`n_cluster`** *(optional)* — Active NLOS clusters per link in `[min(max(1, NumClusters_min - 1), N), N]`;
  omitted = N; `[n_link]`

## Outputs:
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_fd]`, `n_fd = n_freq` for `"38.901_MF"`, else 1
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in [rad], in [-pi, pi); `[n_link, n_slot, n_fd]`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in [rad], in [-pi/2, pi/2]; `[n_link, n_slot, n_fd]`
- **`n_active`** *(optional)* — Number of NLOS clusters with power > 0; `[n_link]`

## See also:
- [[ssf_draw]] (the random values `rnd` and `rnd_freq`)
- [[lsp_generate]] (the LSPs, EoD offset, absolute TOA and ground permittivity)
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
    // "quadriga" places the first NLOS cluster at the absolute TOA; without it the start of the NLOS delays is
    // arbitrary and can come close to the LOS delay
    const bool los = sp.at("KF_mu") > -30.0, toa_def = sp.at("absTOA_mu") > -30.0;
    if (m_qd && !los && toa_def && N > 0 && abs_toa == nullptr)
        throw std::invalid_argument(
            "Method 'quadriga' requires 'abs_toa' in NLOS scenarios that define the absolute TOA ('absTOA_mu' > -30): "
            "the NLOS clusters start at the excess delay of TR 38.901 clause 7.6.9.");
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
        if (n_link > 0 && (arma::any(rx_pos.col(2) < 0.0) || arma::any(tx_pos.col(2) < 0.0)))
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
    const double r_tau = sp.at("r_DS"), zeta = sp.at("LNS_ksi"), r2d = 180.0 / arma::datum::pi;
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
        const double los_ang[4] = {aod_l, qd_wrap(aod_l + arma::datum::pi), eod_l, -eod_l}; // AOD, AOA, EOD, EOA

        // Ground reflection (TR 38.901 clause 7.6.8, ground at z = 0): delay (d_GR - d3D)/c in the cancellation-free
        // form 4·h_t·h_r/((d_GR + d3D)·c), elevation -atan2(h_t + h_r, d2D), the power factor (TR methods
        // (d3D/d_GR)², eq. (7.6-32); "quadriga" 1) and R² = (abs(R_par)² + abs(R_perp)²)/2 per frequency of eqs.
        // (7.6-38), (7.6-39) with eps = re - j·im (eq. (7.6-40)), cos(theta) = -(h_t + h_r)/d_GR' and
        // sin²(theta) = (d2D/d_GR')², d_GR' = max(d_GR, 1e-5); a zero denominator (grazing with eps = 1) gives R = 0
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
            double centre[4];
            for (arma::uword c = 0; c < 9 * N; ++c)
                x[c] = rnd(l, c);
            for (arma::uword k = 0; k < F && !m_qd; ++k)
                for (arma::uword n = 0; n < N; ++n)
                    z[k * N + n] = (*rnd_freq)(l, n, k);
            if (m_tr || m_mf) // LOS direction {AOD, AOA, ZOD, ZOA} in [deg]
                for (arma::uword a = 0; a < 4; ++a)
                    centre[a] = (a < 2) ? los_ang[a] * r2d : 90.0 - los_ang[a] * r2d;
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
                qd_ssf_quadriga(x.data(), N, N_l, F, los, r_tau, spk.data(), kf.data(), los_ang, G, tau_gr, e_gr,
                                R2.data(), delta, c_delay.data(), c_power.data(), c_ang.data());
            }

            // Absolute TOA in NLOS scenarios (TR 38.901 eq. (7.6-43)): the TR methods add it; "quadriga" moves the
            // first NLOS cluster to it (scenarios without absolute TOA: only where the input is > 0)
            if (!los && abs_toa != nullptr)
            {
                const double toa = (*abs_toa)(l);
                const double t0 = m_qd ? *std::min_element(c_delay.begin(), c_delay.end()) : 0.0;
                if (!m_qd || toa_def || toa > 0.0)
                    for (double &d : c_delay)
                        d = d - t0 + toa;
            }
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
