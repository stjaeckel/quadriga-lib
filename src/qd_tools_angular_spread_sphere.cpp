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

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads with spherical wrapping

## Description:
- Calculates the RMS azimuth and elevation angular spreads from a set of power-weighted angles.
- Uses spherical coordinate wrapping to avoid the pole singularity: the power-weighted mean
  direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies
  on the equator before computing spreads.
- Without spherical wrapping, azimuth spread near the poles is inflated (large azimuth spread
  despite energy being focused into a small solid angle). This method corrects for that.
- Optionally computes an optimal bank (roll) angle that maximizes azimuth spread and minimizes
  elevation spread, corresponding to the principal axes of the angular power distribution.
- The bank angle is derived analytically from the eigenvectors of the 2×2 power-weighted
  covariance matrix of the centered azimuth and elevation angles.
- An optional quantization step can group nearby paths before computing the spread.
- If `pow` has only 1 row but `az` has `n_ang` rows, the power vector is replicated for all
  angle sets.
- If `el` has only 1 row but `az` has `n_ang` rows, elevation is assumed to be zero for all paths.

## Declaration:
```
template <typename dtype>
void quadriga_lib::calc_angular_spreads_sphere(
    const arma::Mat<dtype> &az,
    const arma::Mat<dtype> &el,
    const arma::Mat<dtype> &pow,
    arma::Col<dtype> *azimuth_spread = nullptr,
    arma::Col<dtype> *elevation_spread = nullptr,
    arma::Mat<dtype> *orientation = nullptr,
    arma::Mat<dtype> *phi = nullptr,
    arma::Mat<dtype> *theta = nullptr,
    bool calc_bank_angle = true,
    dtype quantize = (dtype)0);
```

## Arguments:
- `const arma::Mat<dtype> &**az**` (input)<br>
  Azimuth angles in [rad], ranging from -pi to pi. Size `[n_ang, n_path]`.

- `const arma::Mat<dtype> &**el**` (input)<br>
  Elevation angles in [rad], ranging from -pi/2 to pi/2. Size `[n_ang, n_path]` or `[1, n_path]`.

- `const arma::Mat<dtype> &**pow**` (input)<br>
  Path powers in [W]. Size `[n_ang, n_path]` or `[1, n_path]`.

- `arma::Col<dtype> ***azimuth_spread** = nullptr` (optional output)<br>
  RMS azimuth angular spread in [rad]. Length `[n_ang]`.

- `arma::Col<dtype> ***elevation_spread** = nullptr` (optional output)<br>
  RMS elevation angular spread in [rad]. Length `[n_ang]`.

- `arma::Mat<dtype> ***orientation** = nullptr` (optional output)<br>
  Power-weighted mean-angle orientation using aircraft principal axes: row 0 = bank angle,
  row 1 = tilt angle, row 2 = heading angle, all in [rad]. Size `[3, n_ang]`.

- `arma::Mat<dtype> ***phi** = nullptr` (optional output)<br>
  Rotated azimuth angles in [rad]. Size `[n_ang, n_path]`.

- `arma::Mat<dtype> ***theta** = nullptr` (optional output)<br>
  Rotated elevation angles in [rad]. Size `[n_ang, n_path]`.

- `bool **calc_bank_angle** = true` (input)<br>
  If true, the optimal bank angle is computed analytically. If false, bank is set to zero.

- `dtype **quantize** = 0` (input)<br>
  Angular quantization step in [deg]. Paths within this angular distance are grouped and their
  powers summed before computing the spread. Set to 0 to treat all paths independently.

## Example:
```
arma::mat az = {0.1, 0.2, -0.1, 0.3};      // 1 angle set, 4 paths
arma::mat el = {0.0, 0.05, -0.05, 0.02};
arma::mat pw = {1.0, 2.0, 1.5, 0.5};

arma::vec as, es;
arma::mat orient;
quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es, &orient);
// as(0) contains the azimuth spread, es(0) the elevation spread
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
        dp[i] = std::atan2(std::sin(d), std::cos(d)); // wrap
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
void quadriga_lib::calc_angular_spreads_sphere(const arma::Mat<dtype> &az,
                                               const arma::Mat<dtype> &el,
                                               const arma::Mat<dtype> &pow,
                                               arma::Col<dtype> *azimuth_spread,
                                               arma::Col<dtype> *elevation_spread,
                                               arma::Mat<dtype> *orientation,
                                               arma::Mat<dtype> *phi_out,
                                               arma::Mat<dtype> *theta_out,
                                               bool calc_bank_angle,
                                               dtype quantize)
{
    arma::uword N = az.n_rows; // Number of angle sets
    arma::uword L = az.n_cols; // Number of paths

    // --- Input validation ---
    if (N == 0 || L == 0)
        throw std::invalid_argument("Input 'az' must not be empty.");

    if (el.n_cols != L)
        throw std::invalid_argument("Input 'el' must have the same number of columns as 'az'.");

    if (pow.n_cols != L)
        throw std::invalid_argument("Input 'pow' must have the same number of columns as 'az'.");

    if (el.n_rows != N && el.n_rows != 1 && el.n_rows != 0)
        throw std::invalid_argument("Input 'el' must have n_ang rows, 1 row, or be empty.");

    if (pow.n_rows != N && pow.n_rows != 1)
        throw std::invalid_argument("Input 'pow' must have n_ang rows or 1 row.");

    bool el_broadcast = (el.n_rows < N);
    bool pow_broadcast = (pow.n_rows < N);

    dtype quantize_rad = quantize * (dtype)(3.14159265358979323846 / 180.0);
    dtype pi_val = (dtype)3.14159265358979323846;

    // --- Allocate outputs ---
    if (azimuth_spread != nullptr)
        azimuth_spread->set_size(N);

    if (elevation_spread != nullptr)
        elevation_spread->set_size(N);

    if (orientation != nullptr)
        orientation->set_size(3, N);

    if (phi_out != nullptr)
        phi_out->set_size(N, L);

    if (theta_out != nullptr)
        theta_out->set_size(N, L);

    // --- Process each angle set ---
    for (arma::uword n = 0; n < N; n++)
    {
        const dtype *az_row = az.colptr(0) + n; // Armadillo is column-major
        const dtype *el_row = el_broadcast ? el.colptr(0) : (el.colptr(0) + n);
        const dtype *pw_row = pow_broadcast ? pow.colptr(0) : (pow.colptr(0) + n);
        arma::uword az_stride = N;
        arma::uword el_stride = el_broadcast ? el.n_rows : N;
        arma::uword pw_stride = pow_broadcast ? pow.n_rows : N;

        // --- Normalize power weights ---
        arma::Col<dtype> pn(L);
        dtype *pnp = pn.memptr();
        dtype pt = (dtype)0;
        for (arma::uword i = 0; i < L; i++)
            pt += pw_row[i * pw_stride];
        if (pt <= (dtype)0)
            throw std::invalid_argument("Sum of powers must be positive.");
        for (arma::uword i = 0; i < L; i++)
            pnp[i] = pw_row[i * pw_stride] / pt;

        // --- Step 1: Closed-form power-weighted mean direction ---
        dtype mx = (dtype)0, my = (dtype)0, mz = (dtype)0;
        for (arma::uword i = 0; i < L; i++)
        {
            dtype a = az_row[i * az_stride];
            dtype e = el_row[i * el_stride];
            dtype ce = std::cos(e);
            mx += pnp[i] * std::cos(a) * ce;
            my += pnp[i] * std::sin(a) * ce;
            mz += pnp[i] * std::sin(e);
        }

        dtype heading = std::atan2(my, mx);
        dtype tilt = std::atan2(mz, std::sqrt(mx * mx + my * my));

        // --- Step 2: Compound rotation Ry(-tilt) * Rz(-heading) ---
        dtype ch = std::cos(heading), sh = std::sin(heading);
        dtype ct = std::cos(tilt), st = std::sin(tilt);

        // R = Ry(-tilt) * Rz(-heading), transposed convention:
        //   row 0: [ ct*ch,   ct*sh,  -st ]
        //   row 1: [ -sh,     ch,      0  ]
        //   row 2: [ st*ch,   st*sh,   ct ]
        dtype R00 = ct * ch, R01 = ct * sh, R02 = -st;
        dtype R10 = -sh, R11 = ch, R12 = (dtype)0;
        dtype R20 = st * ch, R21 = st * sh, R22 = ct;

        // Transform all paths to Cartesian, rotate, convert back to spherical
        arma::Col<dtype> phi_n(L), theta_n(L);
        arma::Col<dtype> Cx(L), Cy(L), Cz(L); // Keep Cartesian for bank-angle rotation
        dtype *phip = phi_n.memptr(), *thp = theta_n.memptr();
        dtype *cxp = Cx.memptr(), *cyp = Cy.memptr(), *czp = Cz.memptr();

        for (arma::uword i = 0; i < L; i++)
        {
            dtype a = az_row[i * az_stride];
            dtype e = el_row[i * el_stride];
            dtype ce = std::cos(e);
            dtype x = std::cos(a) * ce;
            dtype y = std::sin(a) * ce;
            dtype z = std::sin(e);

            // Rotate
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
            // Power-weighted covariance of centered angles
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

            // Principal-axis angle of the 2x2 covariance matrix
            bank = (dtype)0.5 * std::atan2((dtype)-2.0 * cov_pt, var_phi - var_th);

            if (std::abs(bank) > (dtype)1e-8)
            {
                // Apply bank rotation Rx(bank)
                dtype cb = std::cos(bank), sb = std::sin(bank);

                for (arma::uword i = 0; i < L; i++)
                {
                    dtype ry = cb * cyp[i] - sb * czp[i];
                    dtype rz = sb * cyp[i] + cb * czp[i];
                    phip[i] = std::atan2(ry, cxp[i]);
                    thp[i] = std::atan2(rz, std::sqrt(cxp[i] * cxp[i] + ry * ry));
                    // Note: cxp[i] is unchanged by Rx rotation
                    cyp[i] = ry; // Update for potential theta_out
                    czp[i] = rz;
                }

                // Verify: if ES > AS after rotation, swap to the other eigenvector
                dtype as_test = calc_angular_spread_1d(phip, pnp, L, quantize_rad);
                dtype es_test = calc_angular_spread_1d(thp, pnp, L, quantize_rad);

                if (es_test > as_test)
                {
                    // Undo and reapply with bank + pi/2
                    bank = std::atan2(std::sin(bank + pi_val * (dtype)0.5),
                                      std::cos(bank + pi_val * (dtype)0.5));

                    dtype cb2 = std::cos(bank), sb2 = std::sin(bank);

                    // Recompute from original centered Cartesian (before first bank rotation)
                    for (arma::uword i = 0; i < L; i++)
                    {
                        dtype a = az_row[i * az_stride];
                        dtype e = el_row[i * el_stride];
                        dtype ce = std::cos(e);
                        dtype x = std::cos(a) * ce;
                        dtype y = std::sin(a) * ce;
                        dtype z = std::sin(e);

                        // Apply heading+tilt rotation first
                        dtype rx = R00 * x + R01 * y + R02 * z;
                        dtype ry = R10 * x + R11 * y + R12 * z;
                        dtype rz = R20 * x + R21 * y + R22 * z;

                        // Then bank rotation
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

        // --- Store orientation ---
        if (orientation != nullptr)
        {
            dtype *op = orientation->colptr(n);
            op[0] = std::atan2(std::sin(bank), std::cos(bank));
            op[1] = std::atan2(std::sin(tilt), std::cos(tilt));
            op[2] = std::atan2(std::sin(heading), std::cos(heading));
        }

        // --- Store rotated angles ---
        if (phi_out != nullptr)
            for (arma::uword i = 0; i < L; i++)
                phi_out->at(n, i) = phip[i];

        if (theta_out != nullptr)
            for (arma::uword i = 0; i < L; i++)
                theta_out->at(n, i) = thp[i];
    }
}

// --- Explicit template instantiations ---
template void quadriga_lib::calc_angular_spreads_sphere(
    const arma::Mat<float> &az, const arma::Mat<float> &el, const arma::Mat<float> &pow,
    arma::Col<float> *azimuth_spread, arma::Col<float> *elevation_spread,
    arma::Mat<float> *orientation, arma::Mat<float> *phi_out, arma::Mat<float> *theta_out,
    bool calc_bank_angle, float quantize);

template void quadriga_lib::calc_angular_spreads_sphere(
    const arma::Mat<double> &az, const arma::Mat<double> &el, const arma::Mat<double> &pow,
    arma::Col<double> *azimuth_spread, arma::Col<double> *elevation_spread,
    arma::Mat<double> *orientation, arma::Mat<double> *phi_out, arma::Mat<double> *theta_out,
    bool calc_bank_angle, double quantize);
