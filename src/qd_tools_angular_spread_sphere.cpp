// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "quadriga_tools.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads with spherical wrapping

## Description:
- Calculates the RMS azimuth and elevation angular spreads from a set of power-weighted angles.
- Inputs and outputs use `std::vector<arma::Col<dtype>>` so that each channel impulse response
  (CIR) can have a different number of paths.
- Uses spherical coordinate wrapping to avoid the pole singularity: the power-weighted mean
  direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies
  on the equator before computing spreads.
- Without spherical wrapping, azimuth spread near the poles is inflated (large azimuth spread
  despite energy being focused into a small solid angle). This method corrects for that.
- Optionally computes an optimal bank (roll) angle that maximizes azimuth spread and minimizes
  elevation spread, corresponding to the principal axes of the angular power distribution.
- The bank angle is derived analytically from the eigenvectors of the 2x2 power-weighted
  covariance matrix of the centered azimuth and elevation angles.
- An optional quantization step can group nearby paths before computing the spread.
- Setting `disable_wrapping` to true skips the rotation and computes spreads directly from the
  raw azimuth and elevation angles (equivalent to treating them as independent 1D variables).
  In this mode, the orientation output will be zero and phi/theta will equal the input az/el.

## Declaration:
```
template <typename dtype>
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

## Arguments:
- `const std::vector<arma::Col<dtype>> &**az**` (input)<br>
  Azimuth angles in [rad], ranging from -pi to pi. Vector of length `n_cir`, each element has
  length `n_path` (may differ per CIR).

- `const std::vector<arma::Col<dtype>> &**el**` (input)<br>
  Elevation angles in [rad], ranging from -pi/2 to pi/2. Vector of length `n_cir`, each element
  has length `n_path` matching the corresponding element in `az`.

- `const std::vector<arma::Col<dtype>> &**powers**` (input)<br>
  Path powers in [W]. Vector of length `n_cir`, each element has length `n_path` matching the
  corresponding element in `az`.

- `arma::Col<dtype> ***azimuth_spread** = nullptr` (optional output)<br>
  RMS azimuth angular spread in [rad]. Length `[n_cir]`.

- `arma::Col<dtype> ***elevation_spread** = nullptr` (optional output)<br>
  RMS elevation angular spread in [rad]. Length `[n_cir]`.

- `arma::Mat<dtype> ***orientation** = nullptr` (optional output)<br>
  Power-weighted mean-angle orientation using aircraft principal axes: row 0 = bank angle,
  row 1 = tilt angle, row 2 = heading angle, all in [rad]. Size `[3, n_cir]`.

- `std::vector<arma::Col<dtype>> ***phi** = nullptr` (optional output)<br>
  Rotated azimuth angles in [rad]. Vector of length `n_cir`, each element has length `n_path`.

- `std::vector<arma::Col<dtype>> ***theta** = nullptr` (optional output)<br>
  Rotated elevation angles in [rad]. Vector of length `n_cir`, each element has length `n_path`.

- `bool **disable_wrapping** = false` (input)<br>
  If true, skip the spherical rotation and compute spreads directly from raw angles. The
  orientation output will be zero and phi/theta will equal the input az/el.

- `bool **calc_bank_angle** = true` (input)<br>
  If true, the optimal bank angle is computed analytically. Only used when `disable_wrapping`
  is false.

- `dtype **quantize** = 0` (input)<br>
  Angular quantization step in [deg]. Paths within this angular distance are grouped and their
  powers summed before computing the spread. Set to 0 to treat all paths independently.

## Example:
```
std::vector<arma::vec> az(2), el(2), powers(2);
az[0] = {0.1, -0.1, 0.05};              // CIR 0: 3 paths
az[1] = {0.2, -0.2, 0.1, -0.1};         // CIR 1: 4 paths
el[0] = {0.0, 0.0, 0.0};
el[1] = {0.05, -0.05, 0.0, 0.0};
powers[0] = {1.0, 1.0, 0.5};
powers[1] = {2.0, 1.0, 1.5, 0.5};

arma::vec as, es;
arma::mat orient;
quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es, &orient);
// as(0), as(1) contain the azimuth spreads for each CIR
```
MD!*/

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

    // RMS angular spread: sqrt( E[phi^2] - E[phi]^2 )
    dtype E_phi = (dtype)0, E_phi2 = (dtype)0;
    for (arma::uword i = 0; i < L_use; i++)
    {
        E_phi += pw_use[i] * dp_use[i];
        E_phi2 += pw_use[i] * dp_use[i] * dp_use[i];
    }

    dtype var = E_phi2 - E_phi * E_phi;
    return (var > (dtype)0) ? std::sqrt(var) : (dtype)0;
}

// --- Main function ---
template <typename dtype>
void quadriga_lib::calc_angular_spreads_sphere(
    const std::vector<arma::Col<dtype>> &az,
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

// --- Explicit template instantiations ---
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
