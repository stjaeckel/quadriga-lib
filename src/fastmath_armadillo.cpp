// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

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
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- For x in [-pi, pi]: max absolute error = 2^(-22.1); for x in [-500, 500]: 2^(-16.0)
- Either `s` or `c` may be `nullptr` to skip that computation
- Output vectors are resized automatically if needed

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

    const dtype * Xa = x.memptr();
    float * Sa = (s == nullptr) ? nullptr : s->memptr();
    float * Ca = (c == nullptr) ? nullptr : c->memptr();

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
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7); values outside [-1, 1] produce NaN
- Output vector is resized automatically if needed

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

    const dtype * Xa = x.memptr();
    float * Sa = s.memptr();

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
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7); values outside [-1, 1] produce NaN
- Output vector is resized automatically if needed

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

    const dtype * Xa = x.memptr();
    float * Ca = c.memptr();

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
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Returns angles in radians in (-pi, pi]; max error ~3 ULP (~3.6e-7)
- atan2(0, 0) returns 0; atan2(±0, -0) returns ±0 (not ±pi)
- Output vector is resized automatically if needed

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

    const dtype * Ya = y.memptr();
    const dtype * Xa = x.memptr();
    float * Aa = a.memptr();

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
- Interpolates phase via SLERP on normalized directions; amplitudes are linearly interpolated
- Weight `w=0` returns A, `w=1` returns B; per-element weights in [0, 1]
- Near-antipodal inputs (phase difference close to pi) fall back to linear interpolation smoothly
- If both input amplitudes are negligible, output is zero
- Max error vs. double-precision reference: ~5 ULP
- All input vectors must have the same length; output vectors resized automatically
- Output Xr and Xi cannot alias each other
- AVX2-optimized (8 complex pairs/lane); scalar fallback without AVX2

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

    const dtype * pAr = Ar.memptr();
    const dtype * pAi = Ai.memptr();
    const dtype * pBr = Br.memptr();
    const dtype * pBi = Bi.memptr();
    const dtype * pw = w.memptr();
    float * pXr = Xr.memptr();
    float * pXi = Xi.memptr();

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
Convert elementwise azimuth/elevation angles to Cartesian coordinates

## Description:
- Conversion: x = cos(el)*cos(az)*len, y = cos(el)*sin(az)*len, z = sin(el)*len
- Optional pointer outputs `sAZ`, `cAZ`, `sEL`, `cEL` return intermediate sin/cos values; pass `nullptr` to skip
- Output vectors resized automatically if needed
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Precision: GENERIC kernel uses full `dtype` precision (double or float).
- AVX2 kernel always computes in single precision internally — for `dtype=double`, inputs are narrowed to 
  float and results widened back. Use `use_kernel=1` to force GENERIC if full double precision is required.

## Declaration:
```
void fast_geo2cart(
    const arma::Col<dtype> &az,
    const arma::Col<dtype> &el,
    arma::Col<dtype> &x,
    arma::Col<dtype> &y,
    arma::Col<dtype> &z,
    arma::Col<dtype> *sAZ = nullptr,
    arma::Col<dtype> *cAZ = nullptr,
    arma::Col<dtype> *sEL = nullptr,
    arma::Col<dtype> *cEL = nullptr,
    const arma::Col<dtype> *len = nullptr,
    int use_kernel = 0);
```

## Input Arguments:
- **`az`** — Azimuth angles in radians; `[n]`
- **`el`** — Elevation angles in radians; `[n]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x² + y² + z²); `[n]`
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable)

## Output Arguments:
- **`x`** — X-coordinates; `[n]`
- **`y`** — Y-coordinates; `[n]`
- **`z`** — Z-coordinates; `[n]`
- **`sAZ`** *(optional)* — sin(az); `[n]` or `nullptr`
- **`cAZ`** *(optional)* — cos(az); `[n]` or `nullptr`
- **`sEL`** *(optional)* — sin(el); `[n]` or `nullptr`
- **`cEL`** *(optional)* — cos(el); `[n]` or `nullptr`

## See also:
- [[fast_cart2geo]] (inverse conversion)
MD!*/

template <typename dtype>
void quadriga_lib::fast_geo2cart(const arma::Col<dtype> &az, const arma::Col<dtype> &el,
                                 arma::Col<dtype> &x, arma::Col<dtype> &y, arma::Col<dtype> &z,
                                 arma::Col<dtype> *sAZ, arma::Col<dtype> *cAZ,
                                 arma::Col<dtype> *sEL, arma::Col<dtype> *cEL,
                                 const arma::Col<dtype> *len, int use_kernel)
{
    const arma::uword n_val = az.n_elem;

    if (el.n_elem != n_val)
        throw std::invalid_argument("Input vectors 'az' and 'el' must have the same length.");
    if (len && len->n_elem != n_val)
        throw std::invalid_argument("Input vectors 'az', 'el' and 'len' must have the same length.");

    // Kernel selection
#if BUILD_WITH_AVX2
    const bool avx2_ok = runtime_AVX2_Check();
#else
    constexpr bool avx2_ok = false;
#endif

    int kernel = 1; // Default to GENERIC
    if (use_kernel == 1)
        kernel = 1;
    else if (use_kernel == 2)
    {
        if (!avx2_ok)
            throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");
        kernel = 2;
    }
    else // Auto-select (use_kernel == 0)
        kernel = avx2_ok ? 2 : 1;

    // Resize required outputs
    if (x.n_elem != n_val)
        x.set_size(n_val);
    if (y.n_elem != n_val)
        y.set_size(n_val);
    if (z.n_elem != n_val)
        z.set_size(n_val);

    // Resize optional outputs
    if (sAZ && sAZ->n_elem != n_val)
        sAZ->set_size(n_val);
    if (cAZ && cAZ->n_elem != n_val)
        cAZ->set_size(n_val);
    if (sEL && sEL->n_elem != n_val)
        sEL->set_size(n_val);
    if (cEL && cEL->n_elem != n_val)
        cEL->set_size(n_val);

    if (n_val == 0)
        return;

    const dtype * pAZ = az.memptr();
    const dtype * pEL = el.memptr();
    const dtype * pLEN = len ? len->memptr() : nullptr;
    dtype * pX = x.memptr();
    dtype * pY = y.memptr();
    dtype * pZ = z.memptr();
    dtype * pSAZ = sAZ ? sAZ->memptr() : nullptr;
    dtype * pCAZ = cAZ ? cAZ->memptr() : nullptr;
    dtype * pSEL = sEL ? sEL->memptr() : nullptr;
    dtype * pCEL = cEL ? cEL->memptr() : nullptr;

#if BUILD_WITH_AVX2
    if (kernel == 2)
        qd_GEO2CART_AVX2(pAZ, pEL, pLEN, pX, pY, pZ, pSAZ, pCAZ, pSEL, pCEL, n_val);
    else
        qd_GEO2CART_GENERIC(pAZ, pEL, pLEN, pX, pY, pZ, pSAZ, pCAZ, pSEL, pCEL, n_val);
#else
    qd_GEO2CART_GENERIC(pAZ, pEL, pLEN, pX, pY, pZ, pSAZ, pCAZ, pSEL, pCEL, n_val);
#endif
}

template void quadriga_lib::fast_geo2cart(const arma::Col<float> &az, const arma::Col<float> &el,
                                          arma::Col<float> &x, arma::Col<float> &y, arma::Col<float> &z,
                                          arma::Col<float> *sAZ, arma::Col<float> *cAZ,
                                          arma::Col<float> *sEL, arma::Col<float> *cEL,
                                          const arma::Col<float> *len, int use_kernel);

template void quadriga_lib::fast_geo2cart(const arma::Col<double> &az, const arma::Col<double> &el,
                                          arma::Col<double> &x, arma::Col<double> &y, arma::Col<double> &z,
                                          arma::Col<double> *sAZ, arma::Col<double> *cAZ,
                                          arma::Col<double> *sEL, arma::Col<double> *cEL,
                                          const arma::Col<double> *len, int use_kernel);

/*!MD
# fast_cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

## Description:
- Conversion: len = sqrt(x² + y² + z²), az = atan2(y, x), el = asin(clamp(z / len, -1, 1))
- Inputs are arbitrary 3D vectors (not required to be unit-length); `len` returns the Euclidean norm
- z/len is clamped to [-1, 1] before asin to guard against len == 0 and FMA rounding artefacts pushing abs(z/len) slightly above 1
- All inputs must have the same length
- In-place and output-output aliasing not allowed (x/y/z cannot alias az, el, or len; az, el, and len cannot alias each other)
- Output vectors resized automatically if needed
- Allowed datatypes: `float` or `double`
- AVX2 kernel computes internally in single precision (double outputs are cast back from float); GENERIC kernel preserves full `dtype` precision

## Declaration:
```
void quadriga_lib::fast_cart2geo(const arma::fvec &x, const arma::fvec &y, const arma::fvec &z,
                                 arma::fvec &az, arma::fvec &el, arma::fvec *len = nullptr, int use_kernel = 0);

void quadriga_lib::fast_cart2geo(const arma::vec &x, const arma::vec &y, const arma::vec &z,
                                 arma::vec &az, arma::vec &el, arma::vec *len = nullptr, int use_kernel = 0);
```
## Input Arguments:
- **`x`** — X-coordinates; `[n]`
- **`y`** — Y-coordinates; `[n]`
- **`z`** — Z-coordinates; `[n]`
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); default `0`

## Output Arguments:
- **`az`** — Azimuth angles in radians; `[n]`
- **`el`** — Elevation angles in radians; `[n]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x² + y² + z²); `[n]`

## See also:
- [[fast_geo2cart]] (inverse conversion)
MD!*/

template <typename dtype>
void quadriga_lib::fast_cart2geo(const arma::Col<dtype> &x, const arma::Col<dtype> &y, const arma::Col<dtype> &z,
                                 arma::Col<dtype> &az, arma::Col<dtype> &el, arma::Col<dtype> *len, int use_kernel)
{
    const arma::uword n_val = x.n_elem;
    if (y.n_elem != n_val || z.n_elem != n_val)
        throw std::invalid_argument("Input vectors 'x', 'y', and 'z' must have the same length.");

    // Kernel selection
#if BUILD_WITH_AVX2
    const bool avx2_ok = runtime_AVX2_Check();
#else
    constexpr bool avx2_ok = false;
#endif

    int kernel = 1; // Default to GENERIC
    if (use_kernel == 1)
        kernel = 1;
    else if (use_kernel == 2)
    {
        if (!avx2_ok)
            throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");
        kernel = 2;
    }
    else // Auto-select (use_kernel == 0)
        kernel = avx2_ok ? 2 : 1;

    // Resize outputs
    if (az.n_elem != n_val)
        az.set_size(n_val);
    if (el.n_elem != n_val)
        el.set_size(n_val);
    if (len != nullptr && len->n_elem != n_val)
        len->set_size(n_val);

    if (n_val == 0)
        return;

    const dtype * pX = x.memptr();
    const dtype * pY = y.memptr();
    const dtype * pZ = z.memptr();
    dtype * pAZ = az.memptr();
    dtype * pEL = el.memptr();
    dtype * pLEN = len ? len->memptr() : nullptr;

    // Aliasing check
    const void *inputs[] = {pX, pY, pZ};
    const void *outputs[] = {pAZ, pEL, pLEN};
    for (auto iv : inputs)
        for (auto ov : outputs)
            if (ov && iv == ov)
                throw std::invalid_argument("Input and output cannot be the same (in-place operation not allowed).");
    if (pAZ == pEL || (pLEN && (pAZ == pLEN || pEL == pLEN)))
        throw std::invalid_argument("Outputs az, el, and len cannot share buffers.");

#if BUILD_WITH_AVX2
    if (kernel == 2)
        qd_CART2GEO_AVX2(pX, pY, pZ, pAZ, pEL, pLEN, n_val);
    else
        qd_CART2GEO_GENERIC(pX, pY, pZ, pAZ, pEL, pLEN, n_val);
#else
    qd_CART2GEO_GENERIC(pX, pY, pZ, pAZ, pEL, pLEN, n_val);
#endif
}

template void quadriga_lib::fast_cart2geo(const arma::Col<float> &, const arma::Col<float> &, const arma::Col<float> &,
                                          arma::Col<float> &, arma::Col<float> &, arma::Col<float> *, int);
template void quadriga_lib::fast_cart2geo(const arma::Col<double> &, const arma::Col<double> &, const arma::Col<double> &,
                                          arma::Col<double> &, arma::Col<double> &, arma::Col<double> *, int);