// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Array antenna rotations in the test cases, with the Euler angles (bank, tilt, heading) in rad.
// qd_rotate rotates one element (0-based) or, with element = -1, all elements, by arrayant_rotate_pattern;
// qd_rotate_multi does the same for a multi-frequency vector by arrayant_rotate_pattern_multi.
// qd_beamwidth and qd_directivity return the results of arrayant_calc_beamwidth and arrayant_calc_directivity for one
// element (0-based) as scalars; qd_beamwidth computes only the outputs that are not nullptr.

#ifndef QD_TEST_ARRAYANT_HPP
#define QD_TEST_ARRAYANT_HPP

#include "quadriga_lib.hpp"

template <typename dtype>
static inline void qd_rotate(quadriga_lib::arrayant<dtype> &ant, double bank, double tilt = 0.0, double heading = 0.0,
                             unsigned usage = 0, unsigned element = unsigned(-1))
{
    const arma::Col<dtype> orientation = {dtype(bank), dtype(tilt), dtype(heading)};
    const arma::uvec i_element = element == unsigned(-1) ? arma::uvec() : arma::uvec{element};
    quadriga_lib::arrayant_rotate_pattern(ant, orientation, usage, &i_element);
}

template <typename dtype>
static inline void qd_rotate_multi(std::vector<quadriga_lib::arrayant<dtype>> &vec, double bank, double tilt,
                                   double heading, unsigned usage = 0, const arma::uvec &i_element = {})
{
    const arma::Col<dtype> orientation = {dtype(bank), dtype(tilt), dtype(heading)};
    quadriga_lib::arrayant_rotate_pattern_multi(vec, orientation, usage, &i_element);
}

template <typename T> // Keeps the output pointers out of the template argument deduction (nullptr arguments)
struct qd_test_type
{
    using type = T;
};

template <typename dtype>
static inline void qd_beamwidth(const quadriga_lib::arrayant<dtype> &ant, arma::uword element, double threshold_dB,
                                typename qd_test_type<dtype>::type *beamwidth_az = nullptr,
                                typename qd_test_type<dtype>::type *beamwidth_el = nullptr,
                                typename qd_test_type<dtype>::type *az_point_ang = nullptr,
                                typename qd_test_type<dtype>::type *el_point_ang = nullptr)
{
    const arma::uvec i_element = {element};
    dtype *out[] = {beamwidth_az, beamwidth_el, az_point_ang, el_point_ang};
    arma::Col<dtype> res[4];
    quadriga_lib::arrayant_calc_beamwidth(ant, &i_element, dtype(threshold_dB), out[0] ? &res[0] : nullptr,
                                          out[1] ? &res[1] : nullptr, out[2] ? &res[2] : nullptr,
                                          out[3] ? &res[3] : nullptr);
    for (int k = 0; k < 4; ++k)
        if (out[k] != nullptr)
            *out[k] = res[k][0];
}

template <typename dtype>
static inline dtype qd_directivity(const quadriga_lib::arrayant<dtype> &ant, arma::uword element)
{
    const arma::uvec i_element = {element};
    arma::Col<dtype> directivity_dBi;
    quadriga_lib::arrayant_calc_directivity(ant, directivity_dBi, &i_element);
    return directivity_dBi[0];
}

#endif
