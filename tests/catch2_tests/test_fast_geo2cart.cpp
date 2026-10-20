// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "quadriga_math.hpp"
#include "quadriga_lib.hpp"
#include <cmath>

// Kernels to test: 1 = GENERIC, 2 = AVX2 (when available)
static std::vector<int> geo_kernels()
{
    if (quadriga_lib::quadriga_lib_has_AVX2())
        return {1, 2};
    return {1};
}

TEMPLATE_TEST_CASE("fast_geo2cart - Large azimuth angles", "[fast_geo2cart]", float, double)
{
    using dtype = TestType;

    // 9 values: one full block of 8 and a tail of 1
    arma::Col<dtype> az = {(dtype)2.0e9, (dtype)-3.0e9, (dtype)1.0e8, (dtype)5.0e9, (dtype)1.0, (dtype)1.0e10,
                           (dtype)-7.0e7, (dtype)2.5e9, (dtype)2.0e9};
    arma::Col<dtype> el(az.n_elem, arma::fill::value((dtype)0.25));

    for (int k : geo_kernels())
    {
        INFO("use_kernel = " << k);
        arma::Col<dtype> x, y, z;
        quadriga_lib::fast_geo2cart<dtype>(az, el, x, y, z, nullptr, nullptr, nullptr, nullptr, nullptr, k);
        for (arma::uword i = 0; i < az.n_elem; ++i)
        {
            const double a = (double)az[i], e = (double)el[i];
            INFO("az = " << a);
            CHECK(std::abs((double)x[i] - std::cos(e) * std::cos(a)) < 1e-6);
            CHECK(std::abs((double)y[i] - std::cos(e) * std::sin(a)) < 1e-6);
            CHECK(std::abs((double)z[i] - std::sin(e)) < 1e-6);
        }
    }

    // Non-finite azimuth gives NaN for x and y
    arma::Col<dtype> az_nf = {(dtype)INFINITY, (dtype)NAN}, el_nf = {(dtype)0.1, (dtype)0.1};
    for (int k : geo_kernels())
    {
        arma::Col<dtype> x, y, z;
        quadriga_lib::fast_geo2cart<dtype>(az_nf, el_nf, x, y, z, nullptr, nullptr, nullptr, nullptr, nullptr, k);
        for (arma::uword i = 0; i < 2; ++i)
            CHECK((std::isnan(x[i]) && std::isnan(y[i])));
    }
}

TEMPLATE_TEST_CASE("fast_cart2geo - Tiny, huge and non-finite vectors", "[fast_cart2geo]", float, double)
{
    using dtype = TestType;

    // double: components that underflow or overflow when squared in single precision
    // float: small and large values whose squares are still normal single-precision numbers
    const dtype t = std::is_same_v<dtype, float> ? (dtype)1e-15 : (dtype)1e-23;
    const dtype h = std::is_same_v<dtype, float> ? (dtype)1e18 : (dtype)1e20;
    arma::Col<dtype> x = {t, h, (dtype)NAN, (dtype)INFINITY, (dtype)1.0, (dtype)-0.0};
    arma::Col<dtype> y = {t, h, (dtype)1.0, (dtype)INFINITY, (dtype)NAN, (dtype)0.0};
    arma::Col<dtype> z = {t, h, (dtype)1.0, (dtype)0.0, (dtype)0.0, (dtype)0.0};
    const double el_ref = std::asin(1.0 / std::sqrt(3.0)); // 0.6155

    for (int k : geo_kernels())
    {
        INFO("use_kernel = " << k);
        arma::Col<dtype> az, el, len;
        quadriga_lib::fast_cart2geo<dtype>(x, y, z, az, el, &len, k);

        CHECK(std::abs((double)az[0] - M_PI / 4.0) < 1e-6);
        CHECK(std::abs((double)el[0] - el_ref) < 1e-6);
        CHECK(std::abs((double)len[0] / (std::sqrt(3.0) * (double)t) - 1.0) < 1e-6);

        CHECK(std::abs((double)az[1] - M_PI / 4.0) < 1e-6);
        CHECK(std::abs((double)el[1] - el_ref) < 1e-6);
        CHECK(std::abs((double)len[1] / (std::sqrt(3.0) * (double)h) - 1.0) < 1e-6);

        CHECK(std::isnan(az[2])); // NaN x
        CHECK(std::isnan(len[2]));
        CHECK(std::abs((double)az[3] - M_PI / 4.0) < 1e-6); // (inf, inf, 0)
        CHECK(std::isinf(len[3]));
        CHECK(std::isnan(az[4])); // NaN y
        CHECK(std::abs((double)az[5] - M_PI) < 1e-6); // (-0, 0, 0): atan2(+0, -0) = pi
        CHECK(el[5] == (dtype)0.0);
    }
}

TEMPLATE_TEST_CASE("fast_geo2cart / fast_cart2geo - Invalid use_kernel", "[fast_geo2cart]", float, double)
{
    using dtype = TestType;
    arma::Col<dtype> a = {(dtype)1.0, (dtype)0.0}, b = {(dtype)0.0, (dtype)1.0}, c = {(dtype)0.0, (dtype)0.0};
    arma::Col<dtype> o1, o2, o3;

    for (int k : {3, -1, 99, 7})
    {
        INFO("use_kernel = " << k);
        CHECK_THROWS_AS(quadriga_lib::fast_cart2geo<dtype>(a, b, c, o1, o2, &o3, k), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::fast_geo2cart<dtype>(a, b, o1, o2, o3, nullptr, nullptr, nullptr, nullptr, nullptr, k),
                        std::invalid_argument);
    }

    for (int k : {0, 1})
    {
        CHECK_NOTHROW(quadriga_lib::fast_cart2geo<dtype>(a, b, c, o1, o2, &o3, k));
        CHECK_NOTHROW(quadriga_lib::fast_geo2cart<dtype>(a, b, o1, o2, o3, nullptr, nullptr, nullptr, nullptr, nullptr, k));
    }

    // 2 = AVX2 throws when AVX2 is not available (library built without AVX2 or CPU without AVX2)
    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        CHECK_NOTHROW(quadriga_lib::fast_cart2geo<dtype>(a, b, c, o1, o2, &o3, 2));
        CHECK_NOTHROW(quadriga_lib::fast_geo2cart<dtype>(a, b, o1, o2, o3, nullptr, nullptr, nullptr, nullptr, nullptr, 2));
    }
    else
    {
        CHECK_THROWS_AS(quadriga_lib::fast_cart2geo<dtype>(a, b, c, o1, o2, &o3, 2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::fast_geo2cart<dtype>(a, b, o1, o2, o3, nullptr, nullptr, nullptr, nullptr, nullptr, 2),
                        std::invalid_argument);
    }
}
