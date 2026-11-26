// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

// 1D RMS angular spread with wrapping and optional quantization
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
        // quantize_rad >= 1e-3 deg (checked by the caller), so |lo|, |hi| <= 180 000 fit an int
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
Channel data and processing
SECTION!*/

/*!MD
# calc_angular_spread
Calculate the RMS azimuth and elevation angular spreads of each CIR, optionally around the mean direction of its paths

- For one angle, with the path powers $p_l$ normalized to sum 1 over the paths of a CIR (the second-moment definition
  of 3GPP TR 38.901):

  $$\sigma = \sqrt{\sum_l p_l \, \Delta_l^2}, \quad \Delta_l = \arg e^{j (\phi_l - \bar\phi)}$$

  with the circular mean $\bar\phi = \arg \sum_l p_l \, e^{j \phi_l}$
- `wrapping = false`: the spreads of the given azimuth and elevation angles; near the poles, the azimuth spread
  overstates the angular spread
- `wrapping = true`: the paths are first rotated so that their power-weighted mean direction (the mean of the unit
  vectors) lies at azimuth 0 on the equator; `calc_bank_angle` then also rotates them about that direction so that
  the azimuth spread is largest (principal axis of the power-weighted covariance of the rotated angles); `orientation`
  is the direction of an antenna that sees the paths at the rotated angles `phi` and `theta`
- `quantize_deg` > 0: the deviations $\Delta_l$ are rounded to multiples of `quantize_deg` and the powers of equal
  deviations summed, for azimuth and elevation separately

## Declaration:
```
void quadriga_lib::calc_angular_spread(
    const std::vector<arma::Col<dtype>> &az,
    const std::vector<arma::Col<dtype>> &el,
    const std::vector<arma::Col<dtype>> &powers,
    arma::Col<dtype> *azimuth_spread = nullptr,
    arma::Col<dtype> *elevation_spread = nullptr,
    arma::Mat<dtype> *orientation = nullptr,
    std::vector<arma::Col<dtype>> *phi = nullptr,
    std::vector<arma::Col<dtype>> *theta = nullptr,
    bool wrapping = false,
    bool calc_bank_angle = false,
    double quantize_deg = 0.0);
```

## Inputs:
- **`az`** — Path azimuths in rad; `[n_cir]` vectors `[n_path]`, at least one CIR and one path per CIR; `n_path` may
  differ between the CIRs
- **`el`** — Path elevations in rad; `[n_cir]` vectors `[n_path]`
- **`powers`** — Path powers, linear, with a positive sum per CIR; `[n_cir]` vectors `[n_path]`
- **`wrapping`** *(optional)* — Rotate the paths to the equator before computing the spreads
- **`calc_bank_angle`** *(optional)* — Rotate them also about the mean direction, for the largest azimuth
  spread; only with `wrapping`
- **`quantize_deg`** *(optional)* — Quantization step of the angle deviations in deg; 0 = none, else finite and >= 1e-3

## Outputs:
- **`azimuth_spread`** *(optional)* — RMS azimuth spread in rad; `[n_cir]`
- **`elevation_spread`** *(optional)* — RMS elevation spread in rad; `[n_cir]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) in rad of an antenna pointing at the mean
  direction (a positive tilt points down), zero without `wrapping`; `[3, n_cir]`
- **`phi`** *(optional)* — Path azimuths in rad in the frame of that antenna (as [[interpolate_pattern]]
  returns them), the input `az` without `wrapping`; `[n_cir]` vectors `[n_path]`
- **`theta`** *(optional)* — Path elevations in rad in that frame, the input `el` without `wrapping`;
  `[n_cir]` vectors `[n_path]`

## See also:
- [[calc_delay_spread]] (delay spread of the same paths)
- [[interpolate_pattern]] (antenna response in the rotated frame)
MD!*/

template <typename dtype>
void quadriga_lib::calc_angular_spread(const std::vector<arma::Col<dtype>> &az,
                                       const std::vector<arma::Col<dtype>> &el,
                                       const std::vector<arma::Col<dtype>> &powers,
                                       arma::Col<dtype> *azimuth_spread,
                                       arma::Col<dtype> *elevation_spread,
                                       arma::Mat<dtype> *orientation,
                                       std::vector<arma::Col<dtype>> *phi_out,
                                       std::vector<arma::Col<dtype>> *theta_out,
                                       bool wrapping,
                                       bool calc_bank_angle,
                                       double quantize_deg_in)
{
    const dtype quantize_deg = dtype(quantize_deg_in);
    arma::uword N = (arma::uword)az.size();

    // --- Input validation ---
    if (N == 0)
        throw std::invalid_argument("Input 'az' must not be empty.");

    if ((arma::uword)el.size() != N)
        throw std::invalid_argument("Input 'el' must have the same number of elements as 'az'.");

    if ((arma::uword)powers.size() != N)
        throw std::invalid_argument("Input 'powers' must have the same number of elements as 'az'.");

    std::vector<dtype> power_sum(N); // Total power per CIR
    for (arma::uword n = 0; n < N; n++)
    {
        arma::uword L = az[n].n_elem;
        if (L == 0)
            throw std::invalid_argument("Entry " + std::to_string(n) + ": Input 'az' must not be empty.");
        if (el[n].n_elem != L)
            throw std::invalid_argument("Entry " + std::to_string(n) + ": Input 'el' must have the same length as 'az'.");
        if (powers[n].n_elem != L)
            throw std::invalid_argument("Entry " + std::to_string(n) + ": Input 'powers' must have the same length as 'az'.");
        const dtype *pw_ptr = powers[n].memptr();
        dtype pt = (dtype)0;
        for (arma::uword i = 0; i < L; i++)
            pt += pw_ptr[i];
        if (pt <= (dtype)0)
            throw std::invalid_argument("Entry " + std::to_string(n) + ": Sum of 'powers' must be positive.");
        power_sum[n] = pt;
    }

    // quantize_deg: 0 (off) or a finite step >= 1e-3 deg, so that the bin count stays below 360 000 + 2
    if (!(quantize_deg == (dtype)0 || (std::isfinite(quantize_deg) && quantize_deg >= (dtype)1e-3)))
        throw std::invalid_argument("Input 'quantize_deg' must be 0 (off) or a finite value >= 1e-3 deg.");

    qd_check_alias(phi_out, "phi", &az, &el, &powers);
    qd_check_alias(theta_out, "theta", &az, &el, &powers);
    qd_check_alias_outputs(phi_out, "phi", theta_out, "theta");
    qd_check_alias_outputs(azimuth_spread, "azimuth_spread", elevation_spread, "elevation_spread");
    qd_check_alias_outputs(azimuth_spread, "azimuth_spread", orientation, "orientation");
    qd_check_alias_outputs(elevation_spread, "elevation_spread", orientation, "orientation");
    for (arma::uword n = 0; n < N; n++)
        qd_check_alias(azimuth_spread, "azimuth_spread", &az[n], &el[n], &powers[n]),
            qd_check_alias(elevation_spread, "elevation_spread", &az[n], &el[n], &powers[n]),
            qd_check_alias(orientation, "orientation", &az[n], &el[n], &powers[n]);

    dtype quantize_rad = quantize_deg * (dtype)(3.14159265358979323846 / 180.0);
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
        const dtype pt = power_sum[n];
        for (arma::uword i = 0; i < L; i++)
            pnp[i] = pw_ptr[i] / pt;

        // --- Without wrapping: compute 1D spreads directly ---
        if (!wrapping)
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
        dtype elevation = std::atan2(mz, std::sqrt(mx * mx + my * my));

        // --- Step 2: Compound rotation Ry(elevation) * Rz(-heading), moves the mean direction to (1, 0, 0) ---
        dtype ch = std::cos(heading), sh = std::sin(heading);
        dtype ct = std::cos(elevation), st = std::sin(elevation);

        dtype R00 = ct * ch, R01 = ct * sh, R02 = st;
        dtype R10 = -sh, R11 = ch, R12 = (dtype)0;
        dtype R20 = -st * ch, R21 = -st * sh, R22 = ct;

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
            op[0] = std::atan2(-std::sin(bank), std::cos(bank)); // Paths rotated by +bank: the antenna frame by -bank
            op[1] = -elevation; // A positive tilt points down
            op[2] = std::atan2(std::sin(heading), std::cos(heading));
        }

        if (phi_out != nullptr)
            (*phi_out)[n] = phi_n;

        if (theta_out != nullptr)
            (*theta_out)[n] = theta_n;
    }
}

template void quadriga_lib::calc_angular_spread(
    const std::vector<arma::Col<float>> &, const std::vector<arma::Col<float>> &,
    const std::vector<arma::Col<float>> &,
    arma::Col<float> *, arma::Col<float> *, arma::Mat<float> *,
    std::vector<arma::Col<float>> *, std::vector<arma::Col<float>> *,
    bool, bool, double);

template void quadriga_lib::calc_angular_spread(
    const std::vector<arma::Col<double>> &, const std::vector<arma::Col<double>> &,
    const std::vector<arma::Col<double>> &,
    arma::Col<double> *, arma::Col<double> *, arma::Mat<double> *,
    std::vector<arma::Col<double>> *, std::vector<arma::Col<double>> *,
    bool, bool, double);
