// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

// --- Internal helper: 1D RMS angular spread with wrapping and optional quantization ---
template <typename dtype>
static dtype calc_angular_spread_1d(const dtype *ang, const dtype *pw, arma::uword L, dtype quantize_rad)
{
    // Compute wrapped mean angle
    dtype sum_sin = (dtype)0, sum_cos = (dtype)0;
    for (arma::uword i = 0; i < L; i++)
    {
        sum_cos += pw[i] * std::cos(ang[i]);
        sum_sin += pw[i] * std::sin(ang[i]);
    }
    dtype mean_ang = std::atan2(sum_sin, sum_cos);

    // Center and wrap to [-pi, pi]
    arma::Col<dtype> dphi(L);
    dtype *dp = dphi.memptr();
    for (arma::uword i = 0; i < L; i++)
    {
        dtype d = ang[i] - mean_ang;
        dp[i] = std::atan2(std::sin(d), std::cos(d));
    }

    const dtype *pw_use = pw;
    const dtype *dp_use = dp;
    arma::uword L_use = L;

    // Optional quantization
    arma::Col<dtype> edges_vec, powQ_vec;
    if (quantize_rad > (dtype)0)
    {
        const dtype pi_val = (dtype)3.14159265358979323846;
        int lo = (int)std::floor(-pi_val / quantize_rad);
        int hi = (int)std::ceil(pi_val / quantize_rad);

        // Build edge grid, excluding values <= -pi
        std::vector<dtype> edges_tmp;
        for (int k = lo; k <= hi; k++)
        {
            dtype e = (dtype)k * quantize_rad;
            if (e > -pi_val + (dtype)1e-10)
                edges_tmp.push_back(e);
        }

        arma::uword nBins = (arma::uword)edges_tmp.size();
        edges_vec.set_size(nBins);
        powQ_vec.zeros(nBins);
        dtype *edp = edges_vec.memptr();
        dtype *pqp = powQ_vec.memptr();
        for (arma::uword m = 0; m < nBins; m++)
            edp[m] = edges_tmp[m];

        dtype half_q = quantize_rad * (dtype)0.5;
        for (arma::uword m = 0; m < nBins; m++)
        {
            for (arma::uword i = 0; i < L; i++)
            {
                if (m < nBins - 1)
                {
                    if (dp[i] > edp[m] - half_q && dp[i] <= edp[m] + half_q)
                        pqp[m] += pw[i];
                }
                else
                {
                    // Last bin wraps around
                    if (dp[i] > edp[m] - half_q || dp[i] <= edp[0] - half_q)
                        pqp[m] += pw[i];
                }
            }
        }

        dp_use = edp;
        pw_use = pqp;
        L_use = nBins;
    }

    // RMS angular spread: sqrt( sum(pw .* d.^2) ), 3GPP TR 38.901 definition
    dtype E_phi2 = (dtype)0;
    for (arma::uword i = 0; i < L_use; i++)
        E_phi2 += pw_use[i] * dp_use[i] * dp_use[i];

    return (E_phi2 > (dtype)0) ? std::sqrt(E_phi2) : (dtype)0;
}

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads with spherical wrapping

- Computes RMS azimuth and elevation angular spreads from power-weighted angles; each CIR may have a different number of paths.
- RMS spread formula: `sqrt(sum(pw .* d^2))` where `d` are wrapped deviations from the circular mean (3GPP TR 38.901 second-moment definition).
- Mean direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies on the equator before computing spreads, avoiding pole singularity artifacts.
- When `calc_bank_angle = true`, an optimal bank angle maximizing azimuth spread is derived analytically from eigenvectors of the 2x2 power-weighted covariance matrix of centered angles.
- When `disable_wrapping = true`, spreads are computed directly from raw angles; `orientation` will be zero and `phi`/`theta` equal the input `az`/`el`.
- When `quantize > 0`, paths within that angular distance are grouped and their powers summed before computing spreads.

## Declaration:
```
void quadriga_lib::calc_angular_spreads_sphere(
    const std::vector<arma::Col<dtype>> &az,
    const std::vector<arma::Col<dtype>> &el,
    const std::vector<arma::Col<dtype>> &powers,
    arma::Col<dtype> *azimuth_spread = nullptr,
    arma::Col<dtype> *elevation_spread = nullptr,
    arma::Mat<dtype> *orientation = nullptr,
    std::vector<arma::Col<dtype>> *phi = nullptr,
    std::vector<arma::Col<dtype>> *theta = nullptr,
    bool disable_wrapping = false,
    bool calc_bank_angle = true,
    dtype quantize = (dtype)0);
```

## Inputs:
- **`az`** — Azimuth angles; range -pi to pi; `[n_cir]` vector, each element of length `n_path`
- **`el`** — Elevation angles; range -pi/2 to pi/2; same structure as `az`
- **`powers`** — Path powers in [W]; same structure as `az`
- **`disable_wrapping`** *(optional)* — If true, skips spherical rotation and computes spreads from raw angles
- **`calc_bank_angle`** *(optional)* — If true, computes optimal bank angle analytically; only used when `disable_wrapping = false`
- **`quantize`** *(optional)* — Angular quantization step in [deg]; paths within this distance are grouped; 0 disables grouping

## Outputs:
- **`azimuth_spread`** *(optional)* — RMS azimuth spread; `[n_cir]`
- **`elevation_spread`** *(optional)* — RMS elevation spread; `[n_cir]`
- **`orientation`** *(optional)* — Power-weighted mean orientation in Euler angles [bank; tilt; heading]; `[3, n_cir]`
- **`phi`** *(optional)* — Rotated azimuth angles; `[n_cir]` vector, each element of length `n_path`
- **`theta`** *(optional)* — Rotated elevation angles; same structure as `phi`
MD!*/

template <typename dtype>
void quadriga_lib::calc_angular_spreads_sphere(const std::vector<arma::Col<dtype>> &az,
                                               const std::vector<arma::Col<dtype>> &el,
                                               const std::vector<arma::Col<dtype>> &powers,
                                               arma::Col<dtype> *azimuth_spread,
                                               arma::Col<dtype> *elevation_spread,
                                               arma::Mat<dtype> *orientation,
                                               std::vector<arma::Col<dtype>> *phi_out,
                                               std::vector<arma::Col<dtype>> *theta_out,
                                               bool disable_wrapping,
                                               bool calc_bank_angle,
                                               dtype quantize)
{
    arma::uword N = (arma::uword)az.size();

    // --- Input validation ---
    if (N == 0)
        throw std::invalid_argument("Input 'az' must not be empty.");

    if ((arma::uword)el.size() != N)
        throw std::invalid_argument("Input 'el' must have the same number of elements as 'az'.");

    if ((arma::uword)powers.size() != N)
        throw std::invalid_argument("Input 'powers' must have the same number of elements as 'az'.");

    for (arma::uword n = 0; n < N; n++)
    {
        arma::uword L = az[n].n_elem;
        if (L == 0)
            throw std::invalid_argument("Element " + std::to_string(n) + " of 'az' must not be empty.");
        if (el[n].n_elem != L)
            throw std::invalid_argument("Element " + std::to_string(n) + " of 'el' must have the same length as 'az'.");
        if (powers[n].n_elem != L)
            throw std::invalid_argument("Element " + std::to_string(n) + " of 'powers' must have the same length as 'az'.");
    }

    dtype quantize_rad = quantize * (dtype)(3.14159265358979323846 / 180.0);
    dtype pi_val = (dtype)3.14159265358979323846;

    // --- Allocate outputs ---
    if (azimuth_spread != nullptr)
        azimuth_spread->set_size(N);

    if (elevation_spread != nullptr)
        elevation_spread->set_size(N);

    if (orientation != nullptr)
        orientation->zeros(3, N);

    if (phi_out != nullptr)
        phi_out->resize(N);

    if (theta_out != nullptr)
        theta_out->resize(N);

    // --- Process each CIR ---
    for (arma::uword n = 0; n < N; n++)
    {
        arma::uword L = az[n].n_elem;
        const dtype *az_ptr = az[n].memptr();
        const dtype *el_ptr = el[n].memptr();
        const dtype *pw_ptr = powers[n].memptr();

        // --- Normalize power weights ---
        arma::Col<dtype> pn(L);
        dtype *pnp = pn.memptr();
        dtype pt = (dtype)0;
        for (arma::uword i = 0; i < L; i++)
            pt += pw_ptr[i];
        if (pt <= (dtype)0)
            throw std::invalid_argument("Sum of powers for element " + std::to_string(n) + " must be positive.");
        for (arma::uword i = 0; i < L; i++)
            pnp[i] = pw_ptr[i] / pt;

        // --- Disable wrapping: compute 1D spreads directly ---
        if (disable_wrapping)
        {
            if (azimuth_spread != nullptr)
                (*azimuth_spread)(n) = calc_angular_spread_1d(az_ptr, pnp, L, quantize_rad);

            if (elevation_spread != nullptr)
                (*elevation_spread)(n) = calc_angular_spread_1d(el_ptr, pnp, L, quantize_rad);

            if (phi_out != nullptr)
                (*phi_out)[n] = az[n];

            if (theta_out != nullptr)
                (*theta_out)[n] = el[n];

            continue;
        }

        // --- Step 1: Closed-form power-weighted mean direction ---
        dtype mx = (dtype)0, my = (dtype)0, mz = (dtype)0;
        for (arma::uword i = 0; i < L; i++)
        {
            dtype ce = std::cos(el_ptr[i]);
            mx += pnp[i] * std::cos(az_ptr[i]) * ce;
            my += pnp[i] * std::sin(az_ptr[i]) * ce;
            mz += pnp[i] * std::sin(el_ptr[i]);
        }

        dtype heading = std::atan2(my, mx);
        dtype tilt = std::atan2(mz, std::sqrt(mx * mx + my * my));

        // --- Step 2: Compound rotation Ry(-tilt) * Rz(-heading) ---
        dtype ch = std::cos(heading), sh = std::sin(heading);
        dtype ct = std::cos(tilt), st = std::sin(tilt);

        dtype R00 = ct * ch, R01 = ct * sh, R02 = -st;
        dtype R10 = -sh, R11 = ch, R12 = (dtype)0;
        dtype R20 = st * ch, R21 = st * sh, R22 = ct;

        arma::Col<dtype> phi_n(L), theta_n(L);
        arma::Col<dtype> Cx(L), Cy(L), Cz(L);
        dtype *phip = phi_n.memptr(), *thp = theta_n.memptr();
        dtype *cxp = Cx.memptr(), *cyp = Cy.memptr(), *czp = Cz.memptr();

        for (arma::uword i = 0; i < L; i++)
        {
            dtype ce = std::cos(el_ptr[i]);
            dtype x = std::cos(az_ptr[i]) * ce;
            dtype y = std::sin(az_ptr[i]) * ce;
            dtype z = std::sin(el_ptr[i]);

            dtype rx = R00 * x + R01 * y + R02 * z;
            dtype ry = R10 * x + R11 * y + R12 * z;
            dtype rz = R20 * x + R21 * y + R22 * z;

            cxp[i] = rx;
            cyp[i] = ry;
            czp[i] = rz;
            phip[i] = std::atan2(ry, rx);
            thp[i] = std::atan2(rz, std::sqrt(rx * rx + ry * ry));
        }

        // --- Step 3: Optimal bank angle (analytical) ---
        dtype bank = (dtype)0;

        if (calc_bank_angle && L > 1)
        {
            dtype m_phi = (dtype)0, m_th = (dtype)0;
            dtype var_phi = (dtype)0, var_th = (dtype)0, cov_pt = (dtype)0;
            for (arma::uword i = 0; i < L; i++)
            {
                m_phi += pnp[i] * phip[i];
                m_th += pnp[i] * thp[i];
            }
            for (arma::uword i = 0; i < L; i++)
            {
                dtype dp = phip[i] - m_phi;
                dtype dt = thp[i] - m_th;
                var_phi += pnp[i] * dp * dp;
                var_th += pnp[i] * dt * dt;
                cov_pt += pnp[i] * dp * dt;
            }

            bank = (dtype)0.5 * std::atan2((dtype)-2.0 * cov_pt, var_phi - var_th);

            if (std::abs(bank) > (dtype)1e-8)
            {
                dtype cb = std::cos(bank), sb = std::sin(bank);

                for (arma::uword i = 0; i < L; i++)
                {
                    dtype ry = cb * cyp[i] - sb * czp[i];
                    dtype rz = sb * cyp[i] + cb * czp[i];
                    phip[i] = std::atan2(ry, cxp[i]);
                    thp[i] = std::atan2(rz, std::sqrt(cxp[i] * cxp[i] + ry * ry));
                    cyp[i] = ry;
                    czp[i] = rz;
                }

                dtype as_test = calc_angular_spread_1d(phip, pnp, L, quantize_rad);
                dtype es_test = calc_angular_spread_1d(thp, pnp, L, quantize_rad);

                if (es_test > as_test)
                {
                    bank = std::atan2(std::sin(bank + pi_val * (dtype)0.5),
                                      std::cos(bank + pi_val * (dtype)0.5));

                    dtype cb2 = std::cos(bank), sb2 = std::sin(bank);

                    for (arma::uword i = 0; i < L; i++)
                    {
                        dtype ce = std::cos(el_ptr[i]);
                        dtype x = std::cos(az_ptr[i]) * ce;
                        dtype y = std::sin(az_ptr[i]) * ce;
                        dtype z = std::sin(el_ptr[i]);

                        dtype rx = R00 * x + R01 * y + R02 * z;
                        dtype ry = R10 * x + R11 * y + R12 * z;
                        dtype rz = R20 * x + R21 * y + R22 * z;

                        dtype ry2 = cb2 * ry - sb2 * rz;
                        dtype rz2 = sb2 * ry + cb2 * rz;

                        cxp[i] = rx;
                        cyp[i] = ry2;
                        czp[i] = rz2;
                        phip[i] = std::atan2(ry2, rx);
                        thp[i] = std::atan2(rz2, std::sqrt(rx * rx + ry2 * ry2));
                    }
                }
            }
        }

        // --- Step 4: Compute RMS angular spreads ---
        if (azimuth_spread != nullptr)
            (*azimuth_spread)(n) = calc_angular_spread_1d(phip, pnp, L, quantize_rad);

        if (elevation_spread != nullptr)
            (*elevation_spread)(n) = calc_angular_spread_1d(thp, pnp, L, quantize_rad);

        if (orientation != nullptr)
        {
            dtype *op = orientation->colptr(n);
            op[0] = std::atan2(std::sin(bank), std::cos(bank));
            op[1] = std::atan2(std::sin(tilt), std::cos(tilt));
            op[2] = std::atan2(std::sin(heading), std::cos(heading));
        }

        if (phi_out != nullptr)
            (*phi_out)[n] = phi_n;

        if (theta_out != nullptr)
            (*theta_out)[n] = theta_n;
    }
}

template void quadriga_lib::calc_angular_spreads_sphere(
    const std::vector<arma::Col<float>> &, const std::vector<arma::Col<float>> &,
    const std::vector<arma::Col<float>> &,
    arma::Col<float> *, arma::Col<float> *, arma::Mat<float> *,
    std::vector<arma::Col<float>> *, std::vector<arma::Col<float>> *,
    bool, bool, float);

template void quadriga_lib::calc_angular_spreads_sphere(
    const std::vector<arma::Col<double>> &, const std::vector<arma::Col<double>> &,
    const std::vector<arma::Col<double>> &,
    arma::Col<double> *, arma::Col<double> *, arma::Mat<double> *,
    std::vector<arma::Col<double>> *, std::vector<arma::Col<double>> *,
    bool, bool, double);