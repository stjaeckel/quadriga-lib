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

#include "quadriga_lib.hpp"

#if defined(_MSC_VER) // Windows
#include <malloc.h>   // Include for _aligned_malloc and _aligned_free
#endif

#if BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#else // AVX2 disabled
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

// Returns the arrayant_lib version number as a string
#define AUX(x) #x
#define STRINGIFY(x) AUX(x)
std::string quadriga_lib::quadriga_lib_version()
{
    std::string str = STRINGIFY(QUADRIGA_LIB_VERSION);
    std::size_t found = str.find_first_of("_");
    str.replace(found, 1, ".");
    found = str.find_first_of("_");
    str.replace(found, 1, ".");
    str = str.substr(1, str.length());
    return str;
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