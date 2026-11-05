// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of gbsm_rays::solve_scatterers (two-bounce solver): scatterers with exact angles and equal distances,
// the fallback along the ray's own directions, LOS, GR and zero-power rays, validation. Oracles: closed forms and
// the polyline geometry of qd_test_rays.hpp. Cases are added with the implementation.

#include <catch2/catch_test_macros.hpp>
