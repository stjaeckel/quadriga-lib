// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of gbsm_single_link_channel: channel coefficients per snapshot for the mobilities "phase" and "drift", the
// outputs "taps" and "rays", the validity report in 'par', validation. Oracles: get_channels_planar and
// get_channels_spherical with the paths of make_rays (qd_test_rays.hpp). Cases are added with the implementation.

#include <catch2/catch_test_macros.hpp>
