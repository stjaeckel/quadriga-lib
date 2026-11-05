// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of qrt_paths_from_scatterers: path objects with folded path gain and free-space loss, interaction codes,
// round trip through qrt_file_append and qrt_file_read, validation. Oracles: make_rays of qd_test_rays.hpp with
// scatterers. Cases are added with the implementation.

#include <catch2/catch_test_macros.hpp>
