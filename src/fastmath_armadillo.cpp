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

#include "quadriga_math.hpp"
#include "fastmath_vectorized_generic.h"

#if BUILD_WITH_AVX2
#include "fastmath_vectorized_avx2.h"
#include "quadriga_lib_avx2_functions.hpp"
#endif

/*!SECTION
Math Functions
SECTION!*/

/*!MD
# fast_sincos
Compute elementwise approximate sine and/or cosine of a vector

## Description:
- Allowed datatypes: `float` (input `arma::fvec`) or `double` (input `arma::vec`); outputs are always `arma::fvec`
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- For x in [-pi, pi]: max absolute error = 2^(-22.1); for x in [-500, 500]: 2^(-16.0)
- Either `s` or `c` may be `nullptr` to skip that computation
- Output vectors are resized automatically if needed
- OpenMP-parallelized when enabled

## Declaration:
```
void quadriga_lib::fast_sincos(const arma::fvec &x, arma::fvec *s = nullptr, arma::fvec *c = nullptr);
void quadriga_lib::fast_sincos(const arma::vec &x,  arma::fvec *s = nullptr, arma::fvec *c = nullptr);
```

## Input Arguments:
- **`x`** — Input angles in radians; `[n]`

## Output Arguments:
- **`s`** *(optional)* — sin(x); `[n]` or `nullptr`
- **`c`** *(optional)* — cos(x); `[n]` or `nullptr`
MD!*/

template <typename dtype>
void quadriga_lib::fast_sincos(const arma::Col<dtype> &x, arma::fvec *s, arma::fvec *c)
{
    const arma::uword n_val = x.n_elem;

    if (s == nullptr && c == nullptr)
        return;

    if (s != nullptr && s->n_elem != n_val)
        s->set_size(n_val);
    if (c != nullptr && c->n_elem != n_val)
        c->set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *__restrict Xa = x.memptr();
    float *__restrict Sa = (s == nullptr) ? nullptr : s->memptr();
    float *__restrict Ca = (c == nullptr) ? nullptr : c->memptr();

    const void *Xa_v = static_cast<const void *>(Xa);
    const void *Sa_v = static_cast<const void *>(Sa);
    const void *Ca_v = static_cast<const void *>(Ca);

    if ((Sa && Xa_v == Sa_v) || // x and s are the same buffer
        (Ca && Xa_v == Ca_v) || // x and c are the same buffer
        (Sa && Ca && Sa == Ca)) // s and c are the same buffer (same type, direct compare ok)
        throw std::invalid_argument("Input and output cannot be the same (in-place operation not allowed).");

    // --- Compute sin/cos into Sa/Ca ---
#if BUILD_WITH_AVX2
    if (runtime_AVX2_Check())
    {
        if (s == nullptr)
            qd_COS_AVX2(Xa, Ca, n_val);
        else if (c == nullptr)
            qd_SIN_AVX2(Xa, Sa, n_val);
        else
            qd_SINCOS_AVX2(Xa, Sa, Ca, n_val);
    }
    else
    {
        if (s == nullptr)
            qd_COS_GENERIC(Xa, Ca, n_val);
        else if (c == nullptr)
            qd_SIN_GENERIC(Xa, Sa, n_val);
        else
            qd_SINCOS_GENERIC(Xa, Sa, Ca, n_val);
    }
#else
    if (s == nullptr)
        qd_COS_GENERIC(Xa, Ca, n_val);
    else if (c == nullptr)
        qd_SIN_GENERIC(Xa, Sa, n_val);
    else
        qd_SINCOS_GENERIC(Xa, Sa, Ca, n_val);
#endif
}

template void quadriga_lib::fast_sincos(const arma::Col<float> &x, arma::fvec *s, arma::fvec *c);
template void quadriga_lib::fast_sincos(const arma::Col<double> &x, arma::fvec *s, arma::fvec *c);

/*!MD
# fast_asin
Compute elementwise approximate arc-sine of a vector

## Description:
- Allowed datatypes: `float` (input `arma::fvec`) or `double` (input `arma::vec`); output is always `arma::fvec`
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7); values outside [-1, 1] produce NaN
- In-place operation not allowed (input and output cannot alias)
- Output vector is resized automatically if needed
- OpenMP-parallelized when enabled

## Declaration:
```
void quadriga_lib::fast_asin(const arma::fvec &x, arma::fvec &s);
void quadriga_lib::fast_asin(const arma::vec &x,  arma::fvec &s);
```

## Input Arguments:
- **`x`** — Input values in [-1, 1]; `[n]`

## Output Arguments:
- **`s`** — asin(x); `[n]`
MD!*/

template <typename dtype>
void quadriga_lib::fast_asin(const arma::Col<dtype> &x, arma::fvec &s)
{
    const arma::uword n_val = x.n_elem;

    if (s.n_elem != n_val)
        s.set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *__restrict Xa = x.memptr();
    float *__restrict Sa = s.memptr();

    if (static_cast<const void *>(Xa) == static_cast<const void *>(Sa))
        throw std::invalid_argument("Input and output cannot be the same (in-place operation not allowed).");

#if BUILD_WITH_AVX2
    if (runtime_AVX2_Check())
    {
        qd_ASIN_AVX2(Xa, Sa, n_val);
    }
    else
    {
        qd_ASIN_GENERIC(Xa, Sa, n_val);
    }
#else
    qd_ASIN_GENERIC(Xa, Sa, n_val);
#endif
}

template void quadriga_lib::fast_asin(const arma::Col<float> &x, arma::fvec &s);
template void quadriga_lib::fast_asin(const arma::Col<double> &x, arma::fvec &s);

/*!MD
# fast_acos
Compute elementwise approximate arc-cosine of a vector

## Description:
- Allowed datatypes: `float` (input `arma::fvec`) or `double` (input `arma::vec`); output is always `arma::fvec`
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7); values outside [-1, 1] produce NaN
- In-place operation not allowed (input and output cannot alias)
- Output vector is resized automatically if needed
- OpenMP-parallelized when enabled

## Declaration:
```
void quadriga_lib::fast_acos(const arma::fvec &x, arma::fvec &c);
void quadriga_lib::fast_acos(const arma::vec &x,  arma::fvec &c);
```

## Input Arguments:
- **`x`** — Input values in [-1, 1]; `[n]`

## Output Arguments:
- **`c`** — acos(x); `[n]`
MD!*/

template <typename dtype>
void quadriga_lib::fast_acos(const arma::Col<dtype> &x, arma::fvec &c)
{
    const arma::uword n_val = x.n_elem;

    if (c.n_elem != n_val)
        c.set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *__restrict Xa = x.memptr();
    float *__restrict Ca = c.memptr();

    if (static_cast<const void *>(Xa) == static_cast<const void *>(Ca))
        throw std::invalid_argument("Input and output cannot be the same (in-place operation not allowed).");

#if BUILD_WITH_AVX2
    if (runtime_AVX2_Check())
    {
        qd_ACOS_AVX2(Xa, Ca, n_val);
    }
    else
    {
        qd_ACOS_GENERIC(Xa, Ca, n_val);
    }
#else
    qd_ACOS_GENERIC(Xa, Ca, n_val);
#endif
}

template void quadriga_lib::fast_acos(const arma::Col<float> &x, arma::fvec &c);
template void quadriga_lib::fast_acos(const arma::Col<double> &x, arma::fvec &c);

/*!MD
# fast_atan2
Compute elementwise approximate two-argument arc-tangent of two vectors

## Description:
- Allowed datatypes: `float` (input `arma::fvec`) or `double` (input `arma::vec`); output is always `arma::fvec`
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Returns angles in radians in (-pi, pi]; max error ~3 ULP (~3.6e-7)
- atan2(0, 0) returns 0; atan2(±0, -0) returns ±0 (not ±pi)
- Both inputs must have the same length; in-place operation not allowed
- Output vector is resized automatically if needed
- OpenMP-parallelized when enabled

## Declaration:
```
void quadriga_lib::fast_atan2(const arma::fvec &y, const arma::fvec &x, arma::fvec &a);
void quadriga_lib::fast_atan2(const arma::vec &y,  const arma::vec &x,  arma::fvec &a);
```

## Input Arguments:
- **`y`** — Y-coordinates (numerator); `[n]`
- **`x`** — X-coordinates (denominator); `[n]`

## Output Arguments:
- **`a`** — atan2(y, x) in radians; `[n]`
MD!*/

template <typename dtype>
void quadriga_lib::fast_atan2(const arma::Col<dtype> &y, const arma::Col<dtype> &x, arma::fvec &a)
{
    const arma::uword n_val = y.n_elem;

    if (x.n_elem != n_val)
        throw std::invalid_argument("Input vectors 'y' and 'x' must have the same length.");

    if (a.n_elem != n_val)
        a.set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *__restrict Ya = y.memptr();
    const dtype *__restrict Xa = x.memptr();
    float *__restrict Aa = a.memptr();

    // Check for aliasing
    const void *inputs[] = {static_cast<const void *>(Ya), static_cast<const void *>(Xa)};
    for (auto iv : inputs)
        if (iv == static_cast<const void *>(Aa))
            throw std::invalid_argument("Input and output cannot be the same (in-place operation not allowed).");

#if BUILD_WITH_AVX2
    if (runtime_AVX2_Check())
    {
        qd_ATAN2_AVX2(Ya, Xa, Aa, n_val);
    }
    else
    {
        qd_ATAN2_GENERIC(Ya, Xa, Aa, n_val);
    }
#else
    qd_ATAN2_GENERIC(Ya, Xa, Aa, n_val);
#endif
}

template void quadriga_lib::fast_atan2(const arma::Col<float> &y, const arma::Col<float> &x, arma::fvec &a);
template void quadriga_lib::fast_atan2(const arma::Col<double> &y, const arma::Col<double> &x, arma::fvec &a);

/*!MD
# fast_slerp
Compute elementwise approximate SLERP interpolation between two complex-valued vectors

## Description:
- Allowed datatypes: `float` (input `arma::fvec`) or `double` (input `arma::vec`); outputs are always `arma::fvec`
- Interpolates phase via SLERP on normalized directions; amplitudes are linearly interpolated
- Weight `w=0` returns A, `w=1` returns B; per-element weights in [0, 1]
- Near-antipodal inputs (phase difference close to pi) fall back to linear interpolation smoothly
- If both input amplitudes are negligible, output is zero
- Max error vs. double-precision reference: ~5 ULP
- All input vectors must have the same length; output vectors resized automatically
- Output Xr and Xi cannot alias each other
- AVX2-optimized (8 complex pairs/lane); scalar fallback without AVX2
- OpenMP-parallelized when enabled

## Declaration:
```
void quadriga_lib::fast_slerp(const arma::fvec &Ar, const arma::fvec &Ai,
                              const arma::fvec &Br, const arma::fvec &Bi,
                              const arma::fvec &w,
                              arma::fvec &Xr, arma::fvec &Xi);

void quadriga_lib::fast_slerp(const arma::vec &Ar, const arma::vec &Ai,
                              const arma::vec &Br, const arma::vec &Bi,
                              const arma::vec &w,
                              arma::fvec &Xr, arma::fvec &Xi);
```

## Input Arguments:
- **`Ar`** — Real part of source A; `[n]`
- **`Ai`** — Imaginary part of source A; `[n]`
- **`Br`** — Real part of source B; `[n]`
- **`Bi`** — Imaginary part of source B; `[n]`
- **`w`** — Per-element interpolation weight in [0, 1]; `[n]`

## Output Arguments:
- **`Xr`** — Real part of interpolated result; `[n]`
- **`Xi`** — Imaginary part of interpolated result; `[n]`
MD!*/

template <typename dtype>
void quadriga_lib::fast_slerp(const arma::Col<dtype> &Ar, const arma::Col<dtype> &Ai,
                              const arma::Col<dtype> &Br, const arma::Col<dtype> &Bi,
                              const arma::Col<dtype> &w,
                              arma::fvec &Xr, arma::fvec &Xi)
{
    const arma::uword n_val = Ar.n_elem;

    if (Ai.n_elem != n_val || Br.n_elem != n_val || Bi.n_elem != n_val || w.n_elem != n_val)
        throw std::invalid_argument("All input vectors must have the same length.");

    if (Xr.n_elem != n_val)
        Xr.set_size(n_val);
    if (Xi.n_elem != n_val)
        Xi.set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *__restrict pAr = Ar.memptr();
    const dtype *__restrict pAi = Ai.memptr();
    const dtype *__restrict pBr = Br.memptr();
    const dtype *__restrict pBi = Bi.memptr();
    const dtype *__restrict pw = w.memptr();
    float *__restrict pXr = Xr.memptr();
    float *__restrict pXi = Xi.memptr();

    // Check for aliasing between any input and any output
    const void *inputs[] = {pAr, pAi, pBr, pBi, pw};
    const void *outputs[] = {pXr, pXi};
    for (auto iv : inputs)
        for (auto ov : outputs)
            if (iv == ov)
                throw std::invalid_argument("Input and output cannot be the same (in-place operation not allowed).");
    if (pXr == pXi)
        throw std::invalid_argument("Output Xr and Xi cannot be the same buffer.");

#if BUILD_WITH_AVX2
    if (runtime_AVX2_Check())
    {
        qd_SLERP_AVX2(pAr, pAi, pBr, pBi, pw, pXr, pXi, n_val);
    }
    else
    {
        qd_SLERP_GENERIC(pAr, pAi, pBr, pBi, pw, pXr, pXi, n_val);
    }
#else
    qd_SLERP_GENERIC(pAr, pAi, pBr, pBi, pw, pXr, pXi, n_val);
#endif
}

template void quadriga_lib::fast_slerp(const arma::Col<float> &Ar, const arma::Col<float> &Ai,
                                       const arma::Col<float> &Br, const arma::Col<float> &Bi,
                                       const arma::Col<float> &w,
                                       arma::fvec &Xr, arma::fvec &Xi);
template void quadriga_lib::fast_slerp(const arma::Col<double> &Ar, const arma::Col<double> &Ai,
                                       const arma::Col<double> &Br, const arma::Col<double> &Bi,
                                       const arma::Col<double> &w,
                                       arma::fvec &Xr, arma::fvec &Xi);

/*!MD
# fast_geo2cart
Convert elementwise azimuth/elevation angles to unit-sphere Cartesian coordinates

## Description:
- Allowed datatypes: `float` (input `arma::fvec`) or `double` (input `arma::vec`); outputs are always `arma::fvec`
- Conversion: x = cos(el)*cos(az), y = cos(el)*sin(az), z = sin(el)
- Optional pointer outputs `sAZ`, `cAZ`, `sEL`, `cEL` return intermediate sin/cos values; pass `nullptr` to skip
- Both inputs must have the same length; in-place operation not allowed
- Output vectors resized automatically if needed
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- OpenMP-parallelized when enabled

## Declaration:
```
void quadriga_lib::fast_geo2cart(const arma::fvec &az, const arma::fvec &el,
                                 arma::fvec &x, arma::fvec &y, arma::fvec &z,
                                 arma::fvec *sAZ = nullptr, arma::fvec *cAZ = nullptr,
                                 arma::fvec *sEL = nullptr, arma::fvec *cEL = nullptr);

void quadriga_lib::fast_geo2cart(const arma::vec &az, const arma::vec &el,
                                 arma::fvec &x, arma::fvec &y, arma::fvec &z,
                                 arma::fvec *sAZ = nullptr, arma::fvec *cAZ = nullptr,
                                 arma::fvec *sEL = nullptr, arma::fvec *cEL = nullptr);
```

## Input Arguments:
- **`az`** — Azimuth angles in radians; `[n]`
- **`el`** — Elevation angles in radians; `[n]`

## Output Arguments:
- **`x`** — X-coordinates on the unit sphere; `[n]`
- **`y`** — Y-coordinates on the unit sphere; `[n]`
- **`z`** — Z-coordinates on the unit sphere; `[n]`
- **`sAZ`** *(optional)* — sin(az); `[n]` or `nullptr`
- **`cAZ`** *(optional)* — cos(az); `[n]` or `nullptr`
- **`sEL`** *(optional)* — sin(el); `[n]` or `nullptr`
- **`cEL`** *(optional)* — cos(el); `[n]` or `nullptr`

## See also:
- [[fast_cart2geo]] (inverse conversion)
MD!*/

template <typename dtype>
void quadriga_lib::fast_geo2cart(const arma::Col<dtype> &az, const arma::Col<dtype> &el,
                                 arma::fvec &x, arma::fvec &y, arma::fvec &z,
                                 arma::fvec *sAZ, arma::fvec *cAZ,
                                 arma::fvec *sEL, arma::fvec *cEL)
{
    const arma::uword n_val = az.n_elem;

    if (el.n_elem != n_val)
        throw std::invalid_argument("Input vectors 'az' and 'el' must have the same length.");

    // Resize required outputs
    if (x.n_elem != n_val)
        x.set_size(n_val);
    if (y.n_elem != n_val)
        y.set_size(n_val);
    if (z.n_elem != n_val)
        z.set_size(n_val);

    // Resize optional outputs
    if (sAZ != nullptr && sAZ->n_elem != n_val)
        sAZ->set_size(n_val);
    if (cAZ != nullptr && cAZ->n_elem != n_val)
        cAZ->set_size(n_val);
    if (sEL != nullptr && sEL->n_elem != n_val)
        sEL->set_size(n_val);
    if (cEL != nullptr && cEL->n_elem != n_val)
        cEL->set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *__restrict pAZ = az.memptr();
    const dtype *__restrict pEL = el.memptr();
    float *__restrict pX = x.memptr();
    float *__restrict pY = y.memptr();
    float *__restrict pZ = z.memptr();
    float *__restrict pSAZ = (sAZ == nullptr) ? nullptr : sAZ->memptr();
    float *__restrict pCAZ = (cAZ == nullptr) ? nullptr : cAZ->memptr();
    float *__restrict pSEL = (sEL == nullptr) ? nullptr : sEL->memptr();
    float *__restrict pCEL = (cEL == nullptr) ? nullptr : cEL->memptr();

    // Check for aliasing between any input and any output
    const void *inputs[] = {static_cast<const void *>(pAZ), static_cast<const void *>(pEL)};
    const void *outputs[] = {pX, pY, pZ, pSAZ, pCAZ, pSEL, pCEL};
    for (auto iv : inputs)
        for (auto ov : outputs)
            if (ov && iv == ov)
                throw std::invalid_argument("Input and output cannot be the same (in-place operation not allowed).");

#if BUILD_WITH_AVX2
    if (runtime_AVX2_Check())
    {
        qd_GEO2CART_AVX2(pAZ, pEL, pX, pY, pZ, pSAZ, pCAZ, pSEL, pCEL, n_val);
    }
    else
    {
        qd_GEO2CART_GENERIC(pAZ, pEL, pX, pY, pZ, pSAZ, pCAZ, pSEL, pCEL, n_val);
    }
#else
    qd_GEO2CART_GENERIC(pAZ, pEL, pX, pY, pZ, pSAZ, pCAZ, pSEL, pCEL, n_val);
#endif
}

template void quadriga_lib::fast_geo2cart(const arma::Col<float> &az, const arma::Col<float> &el,
                                          arma::fvec &x, arma::fvec &y, arma::fvec &z,
                                          arma::fvec *sAZ, arma::fvec *cAZ,
                                          arma::fvec *sEL, arma::fvec *cEL);
template void quadriga_lib::fast_geo2cart(const arma::Col<double> &az, const arma::Col<double> &el,
                                          arma::fvec &x, arma::fvec &y, arma::fvec &z,
                                          arma::fvec *sAZ, arma::fvec *cAZ,
                                          arma::fvec *sEL, arma::fvec *cEL);

/*!MD
# fast_cart2geo
Convert elementwise unit-sphere Cartesian coordinates to azimuth/elevation angles

## Description:
- Allowed datatypes: `float` (input `arma::fvec`) or `double` (input `arma::vec`); outputs are always `arma::fvec`
- Conversion: az = atan2(y, x), el = asin(clamp(z, -1, 1))
- z is clamped to [-1, 1] before asin to guard against FMA rounding artefacts pushing abs(z) slightly above 1
- All inputs must have the same length
- In-place and output-output aliasing not allowed (x/y/z cannot alias az or el; az and el cannot alias each other)
- Output vectors resized automatically if needed
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- OpenMP-parallelized when enabled

## Declaration:
```
void quadriga_lib::fast_cart2geo(const arma::fvec &x, const arma::fvec &y, const arma::fvec &z,
                                 arma::fvec &az, arma::fvec &el);

void quadriga_lib::fast_cart2geo(const arma::vec &x, const arma::vec &y, const arma::vec &z,
                                 arma::fvec &az, arma::fvec &el);
```

## Input Arguments:
- **`x`** — X-coordinates; `[n]`
- **`y`** — Y-coordinates; `[n]`
- **`z`** — Z-coordinates; `[n]`

## Output Arguments:
- **`az`** — Azimuth angles in radians; `[n]`
- **`el`** — Elevation angles in radians; `[n]`

## See also:
- [[fast_geo2cart]] (inverse conversion)
MD!*/

template <typename dtype>
void quadriga_lib::fast_cart2geo(const arma::Col<dtype> &x, const arma::Col<dtype> &y, const arma::Col<dtype> &z,
                                 arma::fvec &az, arma::fvec &el)
{
    const arma::uword n_val = x.n_elem;

    if (y.n_elem != n_val || z.n_elem != n_val)
        throw std::invalid_argument("Input vectors 'x', 'y', and 'z' must have the same length.");

    if (az.n_elem != n_val)
        az.set_size(n_val);
    if (el.n_elem != n_val)
        el.set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *__restrict pX = x.memptr();
    const dtype *__restrict pY = y.memptr();
    const dtype *__restrict pZ = z.memptr();
    float *__restrict pAZ = az.memptr();
    float *__restrict pEL = el.memptr();

    // Check for aliasing between any input and any output
    const void *inputs[] = {static_cast<const void *>(pX), static_cast<const void *>(pY), static_cast<const void *>(pZ)};
    const void *outputs[] = {static_cast<const void *>(pAZ), static_cast<const void *>(pEL)};
    for (auto iv : inputs)
        for (auto ov : outputs)
            if (iv == ov)
                throw std::invalid_argument("Input and output cannot be the same (in-place operation not allowed).");
    if (pAZ == pEL)
        throw std::invalid_argument("Output az and el cannot be the same buffer.");

#if BUILD_WITH_AVX2
    if (runtime_AVX2_Check())
    {
        qd_CART2GEO_AVX2(pX, pY, pZ, pAZ, pEL, n_val);
    }
    else
    {
        qd_CART2GEO_GENERIC(pX, pY, pZ, pAZ, pEL, n_val);
    }
#else
    qd_CART2GEO_GENERIC(pX, pY, pZ, pAZ, pEL, n_val);
#endif
}

template void quadriga_lib::fast_cart2geo(const arma::Col<float> &x, const arma::Col<float> &y, const arma::Col<float> &z,
                                          arma::fvec &az, arma::fvec &el);
template void quadriga_lib::fast_cart2geo(const arma::Col<double> &x, const arma::Col<double> &y, const arma::Col<double> &z,
                                          arma::fvec &az, arma::fvec &el);