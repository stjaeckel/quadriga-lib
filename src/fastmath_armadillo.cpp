// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_math.hpp"
#include "quadriga_lib_helper_functions.hpp"
#include "fastmath_vectorized_generic.h"

#if BUILD_WITH_AVX2
#include "fastmath_vectorized_avx2.h"
#include "quadriga_lib_avx2_functions.hpp"
#endif

/*!SECTION
Math functions
SECTION!*/

/*!MD
# fast_sincos
Compute elementwise approximate sine and/or cosine of a vector

- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for |x| <= 1e10
  (both kernels; the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; inf and NaN give NaN
- The AVX2 kernel is fastest for |x| <= 2^22 (4.2e6); larger arguments take about 1.5 times as long
- Either `sin_x` or `cos_x` may be `nullptr` to skip that computation
- Outputs are single precision also for double input: the fast_* functions are meant for very large arrays, where a
  double output would add a pass and a copy that doubles the peak memory; double-precision callers use the standard library

## Declaration:
```
void quadriga_lib::fast_sincos(const arma::fvec &x, arma::fvec *sin_x = nullptr, arma::fvec *cos_x = nullptr,
                               int use_kernel = 0);
void quadriga_lib::fast_sincos(const arma::vec &x,  arma::fvec *sin_x = nullptr, arma::fvec *cos_x = nullptr,
                               int use_kernel = 0);
```

## Inputs:
- **`x`** — Input angles; `[n_elem]`
- **`use_kernel`** — Kernel: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2; throws if unavailable; default `0`

## Outputs:
- **`sin_x`** *(optional)* — sin(x); `[n_elem]` or `nullptr`
- **`cos_x`** *(optional)* — cos(x); `[n_elem]` or `nullptr`
MD!*/

template <typename dtype>
void quadriga_lib::fast_sincos(const arma::Col<dtype> &x, arma::fvec *sin_x, arma::fvec *cos_x, int use_kernel)
{
    [[maybe_unused]] const int kernel = qd_kernel_auto_avx2(use_kernel);
    const arma::uword n_val = x.n_elem;

    if (sin_x == nullptr && cos_x == nullptr)
        return;

    qd_check_alias_list({sin_x, cos_x}, {"sin_x", "cos_x"}, &x);

    if (sin_x != nullptr && sin_x->n_elem != n_val)
        sin_x->set_size(n_val);
    if (cos_x != nullptr && cos_x->n_elem != n_val)
        cos_x->set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *Xa = x.memptr();
    float *Sa = (sin_x == nullptr) ? nullptr : sin_x->memptr();
    float *Ca = (cos_x == nullptr) ? nullptr : cos_x->memptr();

#if BUILD_WITH_AVX2
    if (kernel == 2)
    {
        if (Sa == nullptr)
            qd_COS_AVX2(Xa, Ca, n_val);
        else if (Ca == nullptr)
            qd_SIN_AVX2(Xa, Sa, n_val);
        else
            qd_SINCOS_AVX2(Xa, Sa, Ca, n_val);
        return;
    }
#endif
    if (Sa == nullptr)
        qd_COS_GENERIC(Xa, Ca, n_val);
    else if (Ca == nullptr)
        qd_SIN_GENERIC(Xa, Sa, n_val);
    else
        qd_SINCOS_GENERIC(Xa, Sa, Ca, n_val);
}

template void quadriga_lib::fast_sincos(const arma::Col<float> &x, arma::fvec *sin_x, arma::fvec *cos_x, int use_kernel);
template void quadriga_lib::fast_sincos(const arma::Col<double> &x, arma::fvec *sin_x, arma::fvec *cos_x, int use_kernel);

/*!MD
# fast_asin
Compute elementwise approximate arc-sine of a vector

- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7), both kernels; values outside [-1, 1] produce NaN
- Double input is rounded to float first; near ±1 this adds up to 6e-8 / sqrt(1 - x²) (at most 3.5e-4 rad)
- Single-precision output also for double input, as [[fast_sincos]]

## Declaration:
```
void quadriga_lib::fast_asin(const arma::fvec &x, arma::fvec &asin_x, int use_kernel = 0);
void quadriga_lib::fast_asin(const arma::vec &x,  arma::fvec &asin_x, int use_kernel = 0);
```

## Inputs:
- **`x`** — Input values in [-1, 1]; `[n_elem]`
- **`use_kernel`** — Kernel, as in [[fast_sincos]]; default `0`

## Outputs:
- **`asin_x`** — asin(x); `[n_elem]`
MD!*/

template <typename dtype>
void quadriga_lib::fast_asin(const arma::Col<dtype> &x, arma::fvec &asin_x, int use_kernel)
{
    [[maybe_unused]] const int kernel = qd_kernel_auto_avx2(use_kernel);
    const arma::uword n_val = x.n_elem;
    qd_check_alias(&asin_x, "asin_x", &x);

    if (asin_x.n_elem != n_val)
        asin_x.set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *Xa = x.memptr();
    float *Sa = asin_x.memptr();

#if BUILD_WITH_AVX2
    if (kernel == 2)
        return qd_ASIN_AVX2(Xa, Sa, n_val);
#endif
    qd_ASIN_GENERIC(Xa, Sa, n_val);
}

template void quadriga_lib::fast_asin(const arma::Col<float> &x, arma::fvec &asin_x, int use_kernel);
template void quadriga_lib::fast_asin(const arma::Col<double> &x, arma::fvec &asin_x, int use_kernel);

/*!MD
# fast_acos
Compute elementwise approximate arc-cosine of a vector

- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7), both kernels; values outside [-1, 1] produce NaN
- Double input is rounded to float first; near ±1 this adds up to 6e-8 / sqrt(1 - x²) (at most 3.5e-4 rad)
- Single-precision output also for double input, as [[fast_sincos]]

## Declaration:
```
void quadriga_lib::fast_acos(const arma::fvec &x, arma::fvec &acos_x, int use_kernel = 0);
void quadriga_lib::fast_acos(const arma::vec &x,  arma::fvec &acos_x, int use_kernel = 0);
```

## Inputs:
- **`x`** — Input values in [-1, 1]; `[n_elem]`
- **`use_kernel`** — Kernel, as in [[fast_sincos]]; default `0`

## Outputs:
- **`acos_x`** — acos(x); `[n_elem]`
MD!*/

template <typename dtype>
void quadriga_lib::fast_acos(const arma::Col<dtype> &x, arma::fvec &acos_x, int use_kernel)
{
    [[maybe_unused]] const int kernel = qd_kernel_auto_avx2(use_kernel);
    const arma::uword n_val = x.n_elem;
    qd_check_alias(&acos_x, "acos_x", &x);

    if (acos_x.n_elem != n_val)
        acos_x.set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *Xa = x.memptr();
    float *Ca = acos_x.memptr();

#if BUILD_WITH_AVX2
    if (kernel == 2)
        return qd_ACOS_AVX2(Xa, Ca, n_val);
#endif
    qd_ACOS_GENERIC(Xa, Ca, n_val);
}

template void quadriga_lib::fast_acos(const arma::Col<float> &x, arma::fvec &acos_x, int use_kernel);
template void quadriga_lib::fast_acos(const arma::Col<double> &x, arma::fvec &acos_x, int use_kernel);

/*!MD
# fast_atan2
Compute elementwise approximate two-argument arc-tangent of two vectors

- Returns angles in [-pi, pi]; max error ~3 ULP (~3.6e-7), both kernels
- Special values as std::atan2 (both kernels): atan2(±0, +0) = ±0, atan2(±0, -0) = ±pi, atan2(±0, x < 0) = ±pi,
  atan2(±inf, ±inf) = ±pi/4 or ±3pi/4; NaN in x or y gives NaN
- Single-precision output also for double input, as [[fast_sincos]]

## Declaration:
```
void quadriga_lib::fast_atan2(const arma::fvec &y, const arma::fvec &x, arma::fvec &atan2_yx, int use_kernel = 0);
void quadriga_lib::fast_atan2(const arma::vec &y,  const arma::vec &x,  arma::fvec &atan2_yx, int use_kernel = 0);
```

## Inputs:
- **`y`** — Y-coordinates (numerator); `[n_elem]`
- **`x`** — X-coordinates (denominator); `[n_elem]`
- **`use_kernel`** — Kernel, as in [[fast_sincos]]; default `0`

## Outputs:
- **`atan2_yx`** — atan2(y, x); `[n_elem]`
MD!*/

template <typename dtype>
void quadriga_lib::fast_atan2(const arma::Col<dtype> &y, const arma::Col<dtype> &x, arma::fvec &atan2_yx, int use_kernel)
{
    [[maybe_unused]] const int kernel = qd_kernel_auto_avx2(use_kernel);
    const arma::uword n_val = y.n_elem;

    if (x.n_elem != n_val)
        throw std::invalid_argument("Input vectors 'y' and 'x' must have the same length.");
    qd_check_alias(&atan2_yx, "atan2_yx", &y, &x);

    if (atan2_yx.n_elem != n_val)
        atan2_yx.set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *Ya = y.memptr();
    const dtype *Xa = x.memptr();
    float *Aa = atan2_yx.memptr();

#if BUILD_WITH_AVX2
    if (kernel == 2)
        return qd_ATAN2_AVX2(Ya, Xa, Aa, n_val);
#endif
    qd_ATAN2_GENERIC(Ya, Xa, Aa, n_val);
}

template void quadriga_lib::fast_atan2(const arma::Col<float> &y, const arma::Col<float> &x, arma::fvec &atan2_yx, int use_kernel);
template void quadriga_lib::fast_atan2(const arma::Col<double> &y, const arma::Col<double> &x, arma::fvec &atan2_yx, int use_kernel);

/*!MD
# fast_slerp
Compute elementwise approximate SLERP interpolation between two complex-valued vectors

- Interpolates phase via SLERP on normalized directions; amplitudes are linearly interpolated
- Weight `w=0` returns A, `w=1` returns B; per-element weights in [0, 1]
- Near-antipodal inputs (phase difference close to pi) fall back to linear interpolation smoothly
- If both input amplitudes are negligible (below 1.19e-7), output is zero
- Max relative error vs. double-precision reference about 5e-7 for phase differences up to 2 rad, 1.2e-5 up to
  3 rad (both kernels); larger towards the near-antipodal transition to linear interpolation
- The result of an element does not depend on the other elements
- Single-precision output also for double input, as [[fast_sincos]]

## Declaration:
```
void quadriga_lib::fast_slerp(const arma::fvec &A_re, const arma::fvec &A_im,
                              const arma::fvec &B_re, const arma::fvec &B_im,
                              const arma::fvec &w,
                              arma::fvec &X_re, arma::fvec &X_im, int use_kernel = 0);

void quadriga_lib::fast_slerp(const arma::vec &A_re, const arma::vec &A_im,
                              const arma::vec &B_re, const arma::vec &B_im,
                              const arma::vec &w,
                              arma::fvec &X_re, arma::fvec &X_im, int use_kernel = 0);
```

## Inputs:
- **`A_re`** — Real part of source A; `[n_elem]`
- **`A_im`** — Imaginary part of source A; `[n_elem]`
- **`B_re`** — Real part of source B; `[n_elem]`
- **`B_im`** — Imaginary part of source B; `[n_elem]`
- **`w`** — Per-element interpolation weight in [0, 1]; `[n_elem]`
- **`use_kernel`** — Kernel, as in [[fast_sincos]]; default `0`

## Outputs:
- **`X_re`** — Real part of interpolated result; `[n_elem]`
- **`X_im`** — Imaginary part of interpolated result; `[n_elem]`
MD!*/

template <typename dtype>
void quadriga_lib::fast_slerp(const arma::Col<dtype> &A_re, const arma::Col<dtype> &A_im,
                              const arma::Col<dtype> &B_re, const arma::Col<dtype> &B_im,
                              const arma::Col<dtype> &w,
                              arma::fvec &X_re, arma::fvec &X_im, int use_kernel)
{
    [[maybe_unused]] const int kernel = qd_kernel_auto_avx2(use_kernel);
    const arma::uword n_val = A_re.n_elem;

    if (A_im.n_elem != n_val || B_re.n_elem != n_val || B_im.n_elem != n_val || w.n_elem != n_val)
        throw std::invalid_argument("All input vectors must have the same length.");
    qd_check_alias_list({&X_re, &X_im}, {"X_re", "X_im"}, &A_re, &A_im, &B_re, &B_im, &w);

    if (X_re.n_elem != n_val)
        X_re.set_size(n_val);
    if (X_im.n_elem != n_val)
        X_im.set_size(n_val);

    if (n_val == 0)
        return;

    const dtype *pAr = A_re.memptr();
    const dtype *pAi = A_im.memptr();
    const dtype *pBr = B_re.memptr();
    const dtype *pBi = B_im.memptr();
    const dtype *pw = w.memptr();
    float *pXr = X_re.memptr();
    float *pXi = X_im.memptr();

#if BUILD_WITH_AVX2
    if (kernel == 2)
        return qd_SLERP_AVX2(pAr, pAi, pBr, pBi, pw, pXr, pXi, n_val);
#endif
    qd_SLERP_GENERIC(pAr, pAi, pBr, pBi, pw, pXr, pXi, n_val);
}

template void quadriga_lib::fast_slerp(const arma::Col<float> &A_re, const arma::Col<float> &A_im,
                                       const arma::Col<float> &B_re, const arma::Col<float> &B_im,
                                       const arma::Col<float> &w,
                                       arma::fvec &X_re, arma::fvec &X_im, int use_kernel);
template void quadriga_lib::fast_slerp(const arma::Col<double> &A_re, const arma::Col<double> &A_im,
                                       const arma::Col<double> &B_re, const arma::Col<double> &B_im,
                                       const arma::Col<double> &w,
                                       arma::fvec &X_re, arma::fvec &X_im, int use_kernel);

/*!MD
# fast_geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

- Conversion: x = cos(el)*cos(az)*len, y = cos(el)*sin(az)*len, z = sin(el)*len
- Optional pointer outputs `sin_az`, `cos_az`, `sin_el`, `cos_el` return intermediate sin/cos values; pass `nullptr` to skip
- Precision: GENERIC kernel uses full `dtype` precision (double or float).
- AVX2 kernel computes sine and cosine in single precision (max. error about 1.3e-7 for angles up to 1e10;
  `dtype=double` angles are reduced in double precision first), results widened back. Use `use_kernel=1` to force
  GENERIC if full double precision is required.

## Declaration:
```
void fast_geo2cart(
    const arma::Col<dtype> &az,
    const arma::Col<dtype> &el,
    arma::Col<dtype> &x,
    arma::Col<dtype> &y,
    arma::Col<dtype> &z,
    arma::Col<dtype> *sin_az = nullptr,
    arma::Col<dtype> *cos_az = nullptr,
    arma::Col<dtype> *sin_el = nullptr,
    arma::Col<dtype> *cos_el = nullptr,
    const arma::Col<dtype> &len = arma::Col<dtype>(),
    int use_kernel = 0);
```

## Inputs:
- **`az`** — Azimuth angles; `[n_elem]`
- **`el`** — Elevation angles; `[n_elem]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x² + y² + z²); `[n_elem]`
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); other values throw

## Outputs:
- **`x`** — X-coordinates; `[n_elem]`
- **`y`** — Y-coordinates; `[n_elem]`
- **`z`** — Z-coordinates; `[n_elem]`
- **`sin_az`** *(optional)* — sin(az); `[n_elem]` or `nullptr`
- **`cos_az`** *(optional)* — cos(az); `[n_elem]` or `nullptr`
- **`sin_el`** *(optional)* — sin(el); `[n_elem]` or `nullptr`
- **`cos_el`** *(optional)* — cos(el); `[n_elem]` or `nullptr`

## See also:
- [[fast_cart2geo]] (inverse conversion)
MD!*/

template <typename dtype>
void quadriga_lib::fast_geo2cart(const arma::Col<dtype> &az, const arma::Col<dtype> &el,
                                 arma::Col<dtype> &x, arma::Col<dtype> &y, arma::Col<dtype> &z,
                                 arma::Col<dtype> *sin_az, arma::Col<dtype> *cos_az,
                                 arma::Col<dtype> *sin_el, arma::Col<dtype> *cos_el,
                                 const arma::Col<dtype> &len, int use_kernel)
{
    const arma::uword n_val = az.n_elem;

    if (el.n_elem != n_val)
        throw std::invalid_argument("Input vectors 'az' and 'el' must have the same length.");
    if (!len.empty() && len.n_elem != n_val)
        throw std::invalid_argument("Input vectors 'az', 'el' and 'len' must have the same length.");

    [[maybe_unused]] const int kernel = qd_kernel_auto_avx2(use_kernel);

    qd_check_alias_list<arma::Col<dtype>>({&x, &y, &z, sin_az, cos_az, sin_el, cos_el}, {"x", "y", "z", "sin_az", "cos_az", "sin_el", "cos_el"},
                                          &az, &el, &len);

    // Resize required outputs
    if (x.n_elem != n_val)
        x.set_size(n_val);
    if (y.n_elem != n_val)
        y.set_size(n_val);
    if (z.n_elem != n_val)
        z.set_size(n_val);

    // Resize optional outputs
    if (sin_az && sin_az->n_elem != n_val)
        sin_az->set_size(n_val);
    if (cos_az && cos_az->n_elem != n_val)
        cos_az->set_size(n_val);
    if (sin_el && sin_el->n_elem != n_val)
        sin_el->set_size(n_val);
    if (cos_el && cos_el->n_elem != n_val)
        cos_el->set_size(n_val);

    if (n_val == 0)
        return;

    const dtype * pAZ = az.memptr();
    const dtype * pEL = el.memptr();
    const dtype * pLEN = len.empty() ? nullptr : len.memptr();
    dtype * pX = x.memptr();
    dtype * pY = y.memptr();
    dtype * pZ = z.memptr();
    dtype * pSAZ = sin_az ? sin_az->memptr() : nullptr;
    dtype * pCAZ = cos_az ? cos_az->memptr() : nullptr;
    dtype * pSEL = sin_el ? sin_el->memptr() : nullptr;
    dtype * pCEL = cos_el ? cos_el->memptr() : nullptr;

#if BUILD_WITH_AVX2
    if (kernel == 2)
        return qd_GEO2CART_AVX2(pAZ, pEL, pLEN, pX, pY, pZ, pSAZ, pCAZ, pSEL, pCEL, n_val);
#endif
    qd_GEO2CART_GENERIC(pAZ, pEL, pLEN, pX, pY, pZ, pSAZ, pCAZ, pSEL, pCEL, n_val);
}

template void quadriga_lib::fast_geo2cart(const arma::Col<float> &az, const arma::Col<float> &el,
                                          arma::Col<float> &x, arma::Col<float> &y, arma::Col<float> &z,
                                          arma::Col<float> *sin_az, arma::Col<float> *cos_az,
                                          arma::Col<float> *sin_el, arma::Col<float> *cos_el,
                                          const arma::Col<float> &len, int use_kernel);

template void quadriga_lib::fast_geo2cart(const arma::Col<double> &az, const arma::Col<double> &el,
                                          arma::Col<double> &x, arma::Col<double> &y, arma::Col<double> &z,
                                          arma::Col<double> *sin_az, arma::Col<double> *cos_az,
                                          arma::Col<double> *sin_el, arma::Col<double> *cos_el,
                                          const arma::Col<double> &len, int use_kernel);

/*!MD
# fast_cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

- Conversion: len = sqrt(x² + y² + z²), az = atan2(y, x), el = asin(clamp(z / len, -1, 1))
- Inputs are arbitrary 3D vectors (not required to be unit-length); `len` returns the Euclidean norm
- z/len is clamped to [-1, 1] before asin to guard against len == 0 and FMA rounding artefacts pushing abs(z/len) slightly above 1
- All inputs must have the same length
- AVX2 kernel computes the angles in single precision (double outputs are cast back from float); for `dtype=double`,
  `len` and z / len are computed in double precision, so very small or very large vectors keep their direction;
  GENERIC kernel preserves full `dtype` precision
- Azimuth in [-pi, pi] (atan2 special values as in [[fast_atan2]]); NaN input gives NaN in `az`, `el` and `len`

## Declaration:
```
void quadriga_lib::fast_cart2geo(const arma::fvec &x, const arma::fvec &y, const arma::fvec &z,
                                 arma::fvec &az, arma::fvec &el, arma::fvec *len = nullptr, int use_kernel = 0);

void quadriga_lib::fast_cart2geo(const arma::vec &x, const arma::vec &y, const arma::vec &z,
                                 arma::vec &az, arma::vec &el, arma::vec *len = nullptr, int use_kernel = 0);
```
## Inputs:
- **`x`** — X-coordinates; `[n_elem]`
- **`y`** — Y-coordinates; `[n_elem]`
- **`z`** — Z-coordinates; `[n_elem]`
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); other values throw; default `0`

## Outputs:
- **`az`** — Azimuth angles; `[n_elem]`
- **`el`** — Elevation angles; `[n_elem]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x² + y² + z²); `[n_elem]`

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

    [[maybe_unused]] const int kernel = qd_kernel_auto_avx2(use_kernel);

    qd_check_alias_list<arma::Col<dtype>>({&az, &el, len}, {"az", "el", "len"}, &x, &y, &z);

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

#if BUILD_WITH_AVX2
    if (kernel == 2)
        return qd_CART2GEO_AVX2(pX, pY, pZ, pAZ, pEL, pLEN, n_val);
#endif
    qd_CART2GEO_GENERIC(pX, pY, pZ, pAZ, pEL, pLEN, n_val);
}

template void quadriga_lib::fast_cart2geo(const arma::Col<float> &, const arma::Col<float> &, const arma::Col<float> &,
                                          arma::Col<float> &, arma::Col<float> &, arma::Col<float> *, int);
template void quadriga_lib::fast_cart2geo(const arma::Col<double> &, const arma::Col<double> &, const arma::Col<double> &,
                                          arma::Col<double> &, arma::Col<double> &, arma::Col<double> *, int);