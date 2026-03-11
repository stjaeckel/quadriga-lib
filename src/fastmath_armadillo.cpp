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

#include "quadriga_tools.hpp"
#include "fastmath_vectorized_generic.h"

#if BUILD_WITH_AVX2
#include "fastmath_vectorized_avx2.h"
#include "quadriga_lib_avx2_functions.hpp"
#endif

/*!SECTION
Miscellaneous / Tools
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