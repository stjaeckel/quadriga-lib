// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Heap buffer and exceptions for the AVX2 files. This file is compiled without AVX2 flags (see
// quadriga_lib_avx2_functions.hpp).

#include "quadriga_lib_avx2_functions.hpp"

#include <cstdlib>
#include <new>
#include <stdexcept>

void qd_avx2_throw_invalid_argument(const char *message)
{
    throw std::invalid_argument(message);
}

void qd_avx2_throw_bad_alloc()
{
    throw std::bad_alloc();
}

qd_avx2_buffer::qd_avx2_buffer(size_t n, size_t elem_size) : ptr(nullptr)
{
    if (n == 0 || elem_size == 0)
        return;
    ptr = std::calloc(n, elem_size); // Returns nullptr if n * elem_size overflows
    if (ptr == nullptr)
        qd_avx2_throw_bad_alloc();
}

qd_avx2_buffer::~qd_avx2_buffer()
{
    std::free(ptr);
}
