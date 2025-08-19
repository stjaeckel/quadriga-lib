// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

// AVX2 accelerated vectorized math functions
// - Vector lengths must be multiple of 8
// - Input / output vectors do not need to be aligned
// - No sanity checks - incorrect argument formatting leads to undefined behavior

#ifndef quadriga_lib_fastmath_vec_avx2_H
#define quadriga_lib_fastmath_vec_avx2_H

#include <stddef.h>

template <typename dtype> // float or double
void qd_SINCOS_AVX2(const dtype *__restrict x,
                    float *__restrict s,
                    float *__restrict c,
                    size_t n_val); // multiple of 8

template <typename dtype> // float or double
void qd_SIN_AVX2(const dtype *__restrict x,
                 float *__restrict s,
                 size_t n_val); // multiple of 8

template <typename dtype> // float or double
void qd_COS_AVX2(const dtype *__restrict x,
                 float *__restrict c,
                 size_t n_val); // multiple of 8

#endif
