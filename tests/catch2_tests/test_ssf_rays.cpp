// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of ssf_rays with the TR methods "38.901" and "38.901_MF" (additive angle model, TR polarisation
// model): ray layout, coupling, ray delays and taps, angles, Jones matrices of the LOS, GR and NLOS rays, validation.
// Oracles: the transcription of TR 38.901 clause 7.5 steps 7 to 11 in qd_test_rays.hpp; inputs from make_fix and
// make_rnd_ray. Cases are added with the implementation.

#include <catch2/catch_test_macros.hpp>
