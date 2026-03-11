// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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

#ifndef quadriga_lib_slerp_H
#define quadriga_lib_slerp_H

// ------------------------------------------------------------------------------------------------
// Spherical interpolation with linear fallback for a single complex value pair
// ------------------------------------------------------------------------------------------------
template <typename dtype>
void slerp_complex_mf(dtype Ar, dtype Ai, dtype Br, dtype Bi, dtype w, dtype &Xr, dtype &Xi)
{
    constexpr dtype one = dtype(1.0), zero = dtype(0.0), neg_one = dtype(-1.0);
    const dtype R0 = std::numeric_limits<dtype>::epsilon() * std::numeric_limits<dtype>::epsilon() * std::numeric_limits<dtype>::epsilon();
    const dtype R1 = std::numeric_limits<dtype>::epsilon();
    constexpr dtype tL = dtype(-0.999), tS = dtype(-0.99), dT = one / (tS - tL);

    dtype wB = w, wA = one - w;
    dtype ampA = std::sqrt(Ar * Ar + Ai * Ai);
    dtype ampB = std::sqrt(Br * Br + Bi * Bi);

    const bool tinyA = ampA < R1;
    const bool tinyB = ampB < R1;

    if (tinyA && tinyB)
    {
        Xr = zero;
        Xi = zero;
        return;
    }

    dtype gAr = tinyA ? zero : Ar / ampA;
    dtype gAi = tinyA ? zero : Ai / ampA;
    dtype gBr = tinyB ? zero : Br / ampB;
    dtype gBi = tinyB ? zero : Bi / ampB;
    dtype cPhase = (tinyA || tinyB) ? neg_one : gAr * gBr + gAi * gBi;
    bool linear_int = cPhase < tS;

    dtype fXr = zero, fXi = zero;
    if (linear_int)
        fXr = wA * Ar + wB * Br, fXi = wA * Ai + wB * Bi;

    if (cPhase > tL)
    {
        dtype Phase = (cPhase >= one) ? R0 : std::acos(cPhase) + R0;
        dtype sPhase = one / std::sin(Phase);
        dtype wp = std::sin(wB * Phase) * sPhase;
        dtype wn = std::sin(wA * Phase) * sPhase;
        dtype gXr = wn * gAr + wp * gBr;
        dtype gXi = wn * gAi + wp * gBi;
        dtype ampX = wA * ampA + wB * ampB;

        if (linear_int) // Transition zone: blend spherical and linear
        {
            dtype m = (tS - cPhase) * dT, n = one - m;
            fXr = n * gXr * ampX + m * fXr;
            fXi = n * gXi * ampX + m * fXi;
        }
        else
            fXr = gXr * ampX, fXi = gXi * ampX;
    }
    Xr = fXr;
    Xi = fXi;
}

#endif