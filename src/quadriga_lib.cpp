// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"

#if defined(_MSC_VER) // Windows
#include <malloc.h>   // Include for _aligned_malloc and _aligned_free
#endif

#if BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#else // AVX2 disabled
#endif

#if BUILD_WITH_CUDA
#include "quadriga_lib_cuda_functions.hpp"
#else // CUDA disabled
#endif

// Template for time measuring:
// #include <chrono>
//
// Init:
// std::chrono::high_resolution_clock::time_point ts = std::chrono::high_resolution_clock::now(), te;
// uword dur = 0;
//
// Read:
// te = std::chrono::high_resolution_clock::now();
// dur = (uword)std::chrono::duration_cast<std::chrono::nanoseconds>(te - ts).count();
// ts = te;
// std::cout << "A = " << 1.0e-9 * double(dur) << std::endl;

// Returns the quadriga-lib version number as a string
std::string quadriga_lib::quadriga_lib_version()
{
    return QUADRIGA_LIB_VERSION_STR;
}

std::string quadriga_lib::quadriga_lib_armadillo_version()
{
    std::ostringstream versionStream;
    versionStream << ARMA_VERSION_MAJOR << "."
                  << ARMA_VERSION_MINOR << "."
                  << ARMA_VERSION_PATCH;
    return versionStream.str();
}

// Check if AVX2 is supported
bool quadriga_lib::quadriga_lib_has_AVX2()
{
    // Create an aligned dataset
    size_t no_data = 16;
#if defined(_MSC_VER) // Windows
    float *X = (float *)_aligned_malloc(no_data * sizeof(float), 32);
#else // Linux
    float *X = (float *)aligned_alloc(32, no_data * sizeof(float));
#endif

    // Fill
    for (size_t i = 0; i < 16; ++i)
        X[i] = (float)i;

    float Z[8];
    for (size_t i = 0; i < 8; ++i)
        Z[i] = 0.0f;

#if BUILD_WITH_AVX2
    if (runtime_AVX2_Check()) // CPU support for AVX2
    {
        qd_TEST_AVX2(X, Z);
    }
#endif

    // Free aligned memory
#if defined(_MSC_VER) // Windows
    _aligned_free(X);
#else // Linux
    free(X);
#endif

    if (Z[0] == 2.0f && Z[7] == 107.0f)
        return true;

    return false;
}

// Check if CUDA is supported
bool quadriga_lib::quadriga_lib_has_CUDA()
{
#if BUILD_WITH_CUDA
    float X[16];
    for (size_t i = 0; i < 16; ++i)
        X[i] = (float)i;

    float Z[8];
    for (size_t i = 0; i < 8; ++i)
        Z[i] = 0.0f;

    if (runtime_CUDA_Check()) // GPU support for CUDA
    {
        qd_TEST_CUDA(X, Z);
    }

    if (Z[0] == 2.0f && Z[7] == 107.0f)
        return true;
#endif

    return false;
}