// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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
Miscellaneous tools
SECTION!*/

/*!MD
# fast_sincos
Fast, approximate sine/cosine for single-precision vectors

## Description:
Computes elementwise sine and/or cosine for an Armadillo `fvec`. Designed for high throughput on modern CPUs.
- Operates on input angles in radians
- Accepts `arma::fvec` and writes results into provided output pointers
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::sinf` / `std::cosf`
- Either output (`s` or `c`) may be `nullptr` to skip its computation
- Allowed datatype: `float` (Armadillo `fvec`)

## Declaration:
```
void fast_sincos(const arma::fvec &x,
                arma::fvec *s = nullptr,
                arma::fvec *c = nullptr);
```

## Arguments:
- `const arma::fvec **x**` (input)<br>
  Input angles in radians. Size `[n]`.

- `arma::fvec **s** = nullptr` (optional output)<br>
  If non-null, set to `sin(x)`. Resized to length `n` if needed. Size `[n]` or `nullptr`.

- `arma::fvec **c** = nullptr` (optional output)<br>
  If non-null, set to `cos(x)`. Resized to length `n` if needed. Size `[n]` or `nullptr`.

## Returns:
- `void` (output)<br>
  No return value. Results written via output pointers.

## Example:
```
arma::fvec x = arma::linspace[arma::fvec](arma::fvec)(0.0f, 6.2831853f, 1000);
arma::fvec s, c;
fast_sincos(x, &s, &c);      // compute both
fast_sincos(x, &s, nullptr); // compute sine only
fast_sincos(x, nullptr, &c); // compute cosine only
```
MD!*/

void quadriga_lib::fast_sincos(const arma::fvec &x, arma::fvec *s, arma::fvec *c)
{
    const arma::uword n_val = x.n_elem;

    if (s == nullptr && c == nullptr)
        return;

    if (s != nullptr && s->n_elem != n_val)
        s->set_size(n_val);
    if (c != nullptr && c->n_elem != n_val)
        c->set_size(n_val);

    const float *__restrict Xa = x.memptr();
    float *__restrict Sa = (s == nullptr) ? nullptr : s->memptr();
    float *__restrict Ca = (c == nullptr) ? nullptr : c->memptr();

    if (Xa == Sa || Xa == Ca || Sa == Ca)
        throw std::invalid_argument("Input and output cannot be the same (inplace operation not allowed).");

    const size_t bulk = (n_val / 8) * 8; // multiple of 8
    const size_t tail = n_val - bulk;

    // --- Compute sin/cos into Sa/Ca ---
#if BUILD_WITH_AVX2
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
