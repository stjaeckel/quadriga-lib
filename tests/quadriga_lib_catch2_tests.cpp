// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <catch2/catch_session.hpp>

// Include test cases
#include "catch2_tests/test_quadriga_tools.cpp"
#include "catch2_tests/test_qdant_read.cpp"
#include "catch2_tests/test_qdant_write.cpp"
#include "catch2_tests/test_arrayant_interpolate.cpp"
#include "catch2_tests/test_arrayant_combine_pattern.cpp"
#include "catch2_tests/test_arrayant.cpp"
#include "catch2_tests/test_arrayant_obj_export.cpp"
#include "catch2_tests/test_baseband_freq_response.cpp"
#include "catch2_tests/test_arrayant_generate.cpp"
#include "catch2_tests/test_get_channels_spherical.cpp"
#include "catch2_tests/test_get_channels_planar.cpp"
#include "catch2_tests/test_hdf_functions.cpp"
#include "catch2_tests/test_channel.cpp"
#include "catch2_tests/test_ray_point_intersect.cpp"
#include "catch2_tests/test_ray_triangle_intersect.cpp"
#include "catch2_tests/test_ray_mesh_interact.cpp"
#include "catch2_tests/test_calc_diffraction_gain.cpp"
#include "catch2_tests/test_channel_obj_export.cpp"
#include "catch2_tests/test_obj_overlap.cpp"
#include "catch2_tests/test_obj_file_read.cpp"
#include "catch2_tests/test_get_channels_irs.cpp"
#include "catch2_tests/test_point_inside_mesh.cpp"

// Main function to run CATCH2
int main(int argc, char *argv[])
{
  int result = Catch::Session().run(argc, argv);
  return result;
}
