// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_lib_fastmath_vec_avx2_H
#define quadriga_lib_fastmath_vec_avx2_H

#include <stddef.h>

template <typename dtype> // float or double
void qd_SINCOS_AVX2(const dtype *x,
                    float *s,
                    float *c,
                    size_t n_val);

template <typename dtype> // float or double
void qd_SIN_AVX2(const dtype *x,
                 float *s,
                 size_t n_val);

template <typename dtype> // float or double
void qd_COS_AVX2(const dtype *x,
                 float *c,
                 size_t n_val);

template <typename dtype> // float or double
void qd_ASIN_AVX2(const dtype *x,
                  float *s,
                  size_t n_val);

template <typename dtype> // float or double
void qd_ACOS_AVX2(const dtype *x,
                  float *c,
                  size_t n_val);

template <typename dtype> // float or double
void qd_ATAN2_AVX2(const dtype *y,
                   const dtype *x,
                   float *a,
                   size_t n_val);

template <typename dtype> // float or double
void qd_SLERP_AVX2(const dtype *Ar, const dtype *Ai,
                   const dtype *Br, const dtype *Bi,
                   const dtype *w,
                   float *Xr, float *Xi,
                   size_t n_val);

template <typename dtype> // float or double
void qd_GEO2CART_AVX2(const dtype *az, const dtype *el, const dtype *len,
                      dtype *x, dtype *y, dtype *z,
                      dtype *sAZ, dtype *cAZ, dtype *sEL, dtype *cEL,
                      size_t n_val);

template <typename dtype> // float or double
void qd_CART2GEO_AVX2(const dtype *x, const dtype *y, const dtype *z,
                      dtype *az, dtype *el, dtype *len,
                      size_t n_val);
#endif