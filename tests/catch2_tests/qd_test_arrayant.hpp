// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Array antenna rotations in the test cases, with the Euler angles (bank, tilt, heading) in rad.
// qd_rotate rotates one element (0-based) or, with element = -1, all elements, by arrayant_rotate_pattern;
// qd_rotate_multi does the same for a multi-frequency vector by arrayant_rotate_pattern_multi.

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

#endif
