// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"

#if BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#endif

#if BUILD_WITH_CUDA
#include "quadriga_lib_cuda_functions.hpp"
#endif

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# quadriga_lib_version
Version number of quadriga-lib

## Declaration:
```
std::string quadriga_lib::quadriga_lib_version();
```

## Returns:
- Version number in the format `x.y.z`

## See also:
- [[quadriga_lib_has_AVX2]] (AVX2 support of library and CPU)
- [[quadriga_lib_has_CUDA]] (CUDA support of library and GPU)
MD!*/
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

// Random number generator of Armadillo 10.4.0 to 12.6.1 (e.g. Armadillo 10.8.2 on Ubuntu 22.04)
// With a system Armadillo of these versions, its headers declare the generator as an external variable that is
// defined in the Armadillo wrapper library (libarmadillo). quadriga-lib uses Armadillo as headers only and does not
// link that library, so the variable is defined here. Without it, every program that calls arma::randu, arma::randn
// or arma::arma_rng::set_seed and links quadriga-lib but not libarmadillo fails to link (e.g. test_bin).
// Armadillo 12.6.2 and later keep the generator in the headers and never define ARMA_USE_EXTERN_RNG; 10.0 to 10.3
// use another variable. For both the block below is empty.
#if defined(ARMA_USE_EXTERN_RNG) && (ARMA_VERSION_MAJOR > 10 || (ARMA_VERSION_MAJOR == 10 && ARMA_VERSION_MINOR >= 4))
namespace arma
{
    thread_local std::mt19937_64 mt19937_64_instance;
}
#endif

/*!MD
# quadriga_lib_has_AVX2
Check whether AVX2 kernels can be used

- True if the library was built with AVX2 support and the CPU runs the AVX2 test kernel correctly
- The test kernel runs once; later calls return the stored result

## Declaration:
```
bool quadriga_lib::quadriga_lib_has_AVX2();
```

## Returns:
- `true` if AVX2 kernels can be used, else `false`
MD!*/
bool quadriga_lib::quadriga_lib_has_AVX2()
{
    static const bool has_avx2 = []
    {
        float Z[8] = {};

#if BUILD_WITH_AVX2
        if (runtime_AVX2_Check()) // CPU support for AVX2
        {
            alignas(32) float X[16];
            for (size_t i = 0; i < 16; ++i)
                X[i] = (float)i;
            qd_TEST_AVX2(X, Z);
        }
#endif

        return Z[0] == 2.0f && Z[7] == 107.0f;
    }();
    return has_avx2;
}

/*!MD
# quadriga_lib_has_CUDA
Check whether CUDA kernels can be used

- True if the library was built with CUDA support and a CUDA-capable GPU runs the test kernel correctly

## Declaration:
```
bool quadriga_lib::quadriga_lib_has_CUDA();
```

## Returns:
- `true` if CUDA kernels can be used, else `false`
MD!*/
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
