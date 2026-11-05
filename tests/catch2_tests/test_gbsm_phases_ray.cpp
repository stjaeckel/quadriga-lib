// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of the "ray" sets of gbsm_init_phases and gbsm_draw: layout, seeds, mirror map, decorrelation distances
// incl. the ray-delay block, marginal distributions, validation. Oracles: the sum-of-sinusoids evaluation of
// qd_test_sos_oracle.hpp and closed forms. Cases are added with the implementation.

#include <catch2/catch_test_macros.hpp>
