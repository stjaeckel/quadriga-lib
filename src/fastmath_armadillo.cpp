// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (https://sjc-wireless.com)
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
Fast, approximate sine/cosine

## Description:
Computes elementwise sine and/or cosine for an Armadillo vector. Designed for high throughput on modern CPUs.
- Operates on input angles in radians
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::sinf` / `std::cosf`
- For x in [-pi, pi], the maximum absolute error is 2^(-22.1), and larger otherwise
- For x in [-500, 500], the maximum absolute error is 2^(-16.0)
- Either output (`s` or `c`) may be `nullptr` to skip its computation
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

## Declaration:
```
void quadriga_lib::fast_sincos(const arma::fvec &x, arma::fvec *s = nullptr, arma::fvec *c = nullptr);
void quadriga_lib::fast_sincos(const arma::vec &x, arma::fvec *s = nullptr, arma::fvec *c = nullptr);
```

## Arguments:
- `const arma::fvec **x**` or `const arma::vec **x**` (input)<br>
  Input angles in radians. Size `[n]`.

- `arma::fvec ***s** = nullptr` (optional output)<br>
  If non-null, set to `sin(x)`. Resized to length `n` if needed. Size `[n]` or `nullptr`.

- `arma::fvec ***c** = nullptr` (optional output)<br>
  If non-null, set to `cos(x)`. Resized to length `n` if needed. Size `[n]` or `nullptr`.

## Returns:
- `void` (output)<br>
  No return value. Results written via output pointers.

## Example:
```
arma::fvec x = arma::linspace[arma::fvec](arma::fvec)(0.0f, 6.2831853f, 1000);
arma::fvec s, c;
quadriga_lib::fast_sincos(x, &s, &c);      // compute both
quadriga_lib::fast_sincos(x, &s, nullptr); // compute sine only
quadriga_lib::fast_sincos(x, nullptr, &c); // compute cosine only
```
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
Fast, approximate arc-sine

## Description:
Computes elementwise arc-sine for an Armadillo vector. Designed for high throughput on modern CPUs.
- Operates on input values in [-1, 1]
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::asinf`
- For x in [-1, 1], the maximum error is approximately 2 ULP (~2.4e-7)
- Input values outside [-1, 1] produce NaN (IEEE compliant)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

## Declaration:
```
void quadriga_lib::fast_asin(const arma::fvec &x, arma::fvec &s);
void quadriga_lib::fast_asin(const arma::vec &x, arma::fvec &s);
```

## Arguments:
- `const arma::fvec &**x**` or `const arma::vec &**x**` (input)<br>
  Input values in [-1, 1]. Length `[n]`.

- `arma::fvec &**s**` (output)<br>
  Set to `asin(x)`. Resized to length `n` if needed. Length `[n]`.

## Example:
```
arma::fvec x = arma::linspace<arma::fvec>(-1.0f, 1.0f, 1000);
arma::fvec s;
quadriga_lib::fast_asin(x, s);
```
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
Fast, approximate arc-cosine

## Description:
Computes elementwise arc-cosine for an Armadillo vector. Designed for high throughput on modern CPUs.
- Operates on input values in [-1, 1]
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::acosf`
- For x in [-1, 1], the maximum error is approximately 2 ULP (~2.4e-7)
- Input values outside [-1, 1] produce NaN (IEEE compliant)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

## Declaration:
```
void quadriga_lib::fast_acos(const arma::fvec &x, arma::fvec &c);
void quadriga_lib::fast_acos(const arma::vec &x, arma::fvec &c);
```

## Arguments:
- `const arma::fvec &**x**` or `const arma::vec &**x**` (input)<br>
  Input values in [-1, 1]. Length `[n]`.

- `arma::fvec &**c**` (output)<br>
  Set to `acos(x)`. Resized to length `n` if needed. Length `[n]`.

## Example:
```
arma::fvec x = arma::linspace<arma::fvec>(-1.0f, 1.0f, 1000);
arma::fvec c;
quadriga_lib::fast_acos(x, c);
```
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
Fast, approximate two-argument arc-tangent

## Description:
Computes elementwise `atan2(y, x)` for two Armadillo vectors. Designed for high throughput on modern CPUs.
- Returns angles in radians in the range (-pi, pi]
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::atan2f`
- Maximum error is approximately 3 ULP (~3.6e-7) across the full domain
- `atan2(0, 0)` returns 0; `atan2(±0, -0)` returns `±0` (not `±pi`)
- Both input vectors must have the same length
- Input and output cannot alias (in-place operation not allowed)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

## Declaration:
```
void quadriga_lib::fast_atan2(const arma::fvec &y, const arma::fvec &x, arma::fvec &a);
void quadriga_lib::fast_atan2(const arma::vec &y, const arma::vec &x, arma::fvec &a);
```

## Arguments:
- `const arma::fvec &**y**` or `const arma::vec &**y**` (input)<br>
  Y-coordinates (numerator of atan2). Length `[n]`.

- `const arma::fvec &**x**` or `const arma::vec &**x**` (input)<br>
  X-coordinates (denominator of atan2). Length `[n]`.

- `arma::fvec &**a**` (output)<br>
  Set to `atan2(y, x)` in radians. Resized to length `n` if needed. Length `[n]`.

## Example:
```
arma::fvec y = {1.0f, -1.0f, 0.0f, 1.0f};
arma::fvec x = {1.0f,  1.0f, -1.0f, 0.0f};
arma::fvec a;
quadriga_lib::fast_atan2(y, x, a);
// a ≈ {0.7854, -0.7854, 3.1416, 1.5708}
```
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
Fast, approximate spherical interpolation (SLERP) for complex value pairs

## Description:
Interpolates elementwise between two complex-valued vectors using spherical linear interpolation
(SLERP) on the normalised directions and linear interpolation of amplitudes.
- Processes per-element interpolation weights (0 = A, 1 = B)
- AVX2-optimized (8 complex pairs per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Near-antipodal inputs (phase angle close to pi) smoothly transition to a linear fallback
- If both input amplitudes are negligible, the output is zero
- Maximum error versus double-precision reference is approximately 5 ULP
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)
- All input vectors must have the same length

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

## Arguments:
- `const arma::fvec &**Ar**` or `const arma::vec &**Ar**` (input)<br>
  Real part of source A. Length `[n]`.

- `const arma::fvec &**Ai**` or `const arma::vec &**Ai**` (input)<br>
  Imaginary part of source A. Length `[n]`.

- `const arma::fvec &**Br**` or `const arma::vec &**Br**` (input)<br>
  Real part of source B. Length `[n]`.

- `const arma::fvec &**Bi**` or `const arma::vec &**Bi**` (input)<br>
  Imaginary part of source B. Length `[n]`.

- `const arma::fvec &**w**` or `const arma::vec &**w**` (input)<br>
  Per-element interpolation weight in [0, 1]. 0 returns A, 1 returns B. Length `[n]`.

- `arma::fvec &**Xr**` (output)<br>
  Real part of interpolated result. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec &**Xi**` (output)<br>
  Imaginary part of interpolated result. Resized to length `n` if needed. Length `[n]`.

## Returns:
- `void` (output)<br>
  No return value. Results written to Xr and Xi.

## Example:
```
arma::fvec Ar = {1.0f, 0.0f}, Ai = {0.0f, 1.0f};
arma::fvec Br = {0.0f, 1.0f}, Bi = {1.0f, 0.0f};
arma::fvec w = {0.5f, 0.5f};
arma::fvec Xr, Xi;
quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);
```
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
Fast, approximate geographic-to-Cartesian conversion

## Description:
Converts elementwise azimuth/elevation angles (in radians) to unit-sphere Cartesian coordinates.
- x = cos(el) * cos(az), y = cos(el) * sin(az), z = sin(el)
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::sinf` / `std::cosf`
- Optionally returns intermediate sin/cos values via pointer arguments; pass `nullptr` to skip
- Both input vectors must have the same length
- Input and output cannot alias (in-place operation not allowed)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

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

## Arguments:
- `const arma::fvec &**az**` or `const arma::vec &**az**` (input)<br>
  Azimuth angles in radians. Length `[n]`.

- `const arma::fvec &**el**` or `const arma::vec &**el**` (input)<br>
  Elevation angles in radians. Length `[n]`.

- `arma::fvec &**x**` (output)<br>
  X-coordinates on the unit sphere. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec &**y**` (output)<br>
  Y-coordinates on the unit sphere. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec &**z**` (output)<br>
  Z-coordinates on the unit sphere. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec ***sAZ** = nullptr` (optional output)<br>
  If non-null, set to `sin(az)`. Resized to length `n` if needed. Length `[n]` or `nullptr`.

- `arma::fvec ***cAZ** = nullptr` (optional output)<br>
  If non-null, set to `cos(az)`. Resized to length `n` if needed. Length `[n]` or `nullptr`.

- `arma::fvec ***sEL** = nullptr` (optional output)<br>
  If non-null, set to `sin(el)`. Resized to length `n` if needed. Length `[n]` or `nullptr`.

- `arma::fvec ***cEL** = nullptr` (optional output)<br>
  If non-null, set to `cos(el)`. Resized to length `n` if needed. Length `[n]` or `nullptr`.

## Example:
```
arma::fvec az = {0.0f, 1.5708f, 3.1416f};
arma::fvec el = {0.0f, 0.5f, -0.5f};
arma::fvec x, y, z, sAZ, cEL;
quadriga_lib::fast_geo2cart(az, el, x, y, z, &sAZ, nullptr, nullptr, &cEL);
```
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
Fast, approximate Cartesian-to-geographic conversion

## Description:
Converts elementwise unit-sphere Cartesian coordinates to azimuth/elevation angles (in radians).
- az = atan2(y, x), el = asin(clamp(z, -1, 1))
- The z-coordinate is clamped to [-1, 1] before computing asin, guarding against FMA rounding artefacts
  from upstream matrix multiplications that can push abs(z) slightly above 1
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::atan2f` / `std::asinf`
- All input vectors must have the same length
- Input and output cannot alias (in-place operation not allowed)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

## Declaration:
```
void quadriga_lib::fast_cart2geo(const arma::fvec &x, const arma::fvec &y, const arma::fvec &z,
                                 arma::fvec &az, arma::fvec &el);
void quadriga_lib::fast_cart2geo(const arma::vec &x, const arma::vec &y, const arma::vec &z,
                                 arma::fvec &az, arma::fvec &el);
```

## Arguments:
- `const arma::fvec &**x**` or `const arma::vec &**x**` (input)<br>
  X-coordinates. Length `[n]`.

- `const arma::fvec &**y**` or `const arma::vec &**y**` (input)<br>
  Y-coordinates. Length `[n]`.

- `const arma::fvec &**z**` or `const arma::vec &**z**` (input)<br>
  Z-coordinates. Length `[n]`.

- `arma::fvec &**az**` (output)<br>
  Azimuth angles in radians. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec &**el**` (output)<br>
  Elevation angles in radians. Resized to length `n` if needed. Length `[n]`.

## Example:
```
arma::fvec x = {1.0f, 0.0f, -1.0f};
arma::fvec y = {0.0f, 1.0f, 0.0f};
arma::fvec z = {0.0f, 0.0f, 0.0f};
arma::fvec az, el;
quadriga_lib::fast_cart2geo(x, y, z, az, el);
// az ≈ {0.0, 1.5708, 3.1416},  el ≈ {0.0, 0.0, 0.0}
```
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