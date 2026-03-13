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
    const size_t bulk = (n_val / 8) * 8; // multiple of 8
    const size_t tail = n_val - bulk;

    if (runtime_AVX2_Check())
    {
        if (bulk)
        {
            if (s == nullptr)
                qd_COS_AVX2(Xa, Ca, bulk);
            else if (c == nullptr)
                qd_SIN_AVX2(Xa, Sa, bulk);
            else
                qd_SINCOS_AVX2(Xa, Sa, Ca, bulk);
        }
        if (tail)
        {
            if (s == nullptr)
                qd_COS_GENERIC(Xa + bulk, Ca + bulk, tail);
            else if (c == nullptr)
                qd_SIN_GENERIC(Xa + bulk, Sa + bulk, tail);
            else
                qd_SINCOS_GENERIC(Xa + bulk, Sa + bulk, Ca + bulk, tail);
        }
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

/*!SECTION
Miscellaneous / Tools
SECTION!*/

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
    const size_t bulk = (n_val / 8) * 8;
    const size_t tail = n_val - bulk;

    if (runtime_AVX2_Check())
    {
        if (bulk)
            qd_ASIN_AVX2(Xa, Sa, bulk);
        if (tail)
            qd_ASIN_GENERIC(Xa + bulk, Sa + bulk, tail);
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

/*!SECTION
Miscellaneous / Tools
SECTION!*/

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
    const size_t bulk = (n_val / 8) * 8;
    const size_t tail = n_val - bulk;

    if (runtime_AVX2_Check())
    {
        if (bulk)
            qd_ACOS_AVX2(Xa, Ca, bulk);
        if (tail)
            qd_ACOS_GENERIC(Xa + bulk, Ca + bulk, tail);
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

/*!SECTION
Miscellaneous / Tools
SECTION!*/

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
    const size_t bulk = (n_val / 8) * 8;
    const size_t tail = n_val - bulk;

    if (runtime_AVX2_Check())
    {
        if (bulk)
            qd_ATAN2_AVX2(Ya, Xa, Aa, bulk);
        if (tail)
            qd_ATAN2_GENERIC(Ya + bulk, Xa + bulk, Aa + bulk, tail);
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

/*!SECTION
Miscellaneous / Tools
SECTION!*/

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
- Maximum error versus double-precision reference is approximately 100 ULP (~1.2e-5 relative error, ~17.5 effective bits)
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
    const size_t bulk = (n_val / 8) * 8;
    const size_t tail = n_val - bulk;

    if (runtime_AVX2_Check())
    {
        if (bulk)
            qd_SLERP_AVX2(pAr, pAi, pBr, pBi, pw, pXr, pXi, bulk);
        if (tail)
            qd_SLERP_GENERIC(pAr + bulk, pAi + bulk, pBr + bulk, pBi + bulk, pw + bulk,
                              pXr + bulk, pXi + bulk, tail);
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