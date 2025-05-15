// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "quadriga_tools.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# generate_diffraction_paths
Generate propagation paths for estimating the diffraction gain

## Description:
This function generates the elliptic propagation paths and corresponding weights necessary for the
calculation of the diffraction gain in <a href="#calc_diffraction_gain">calc_diffraction_gain</a>.

## Caveat:
- Each ellipsoid consists of `n_path` diffraction paths. The number of paths is determined by the
  level of detail (`lod`).
- All diffraction paths of an ellipsoid originate at `orig` and arrive at `dest`
- Each diffraction path has `n_seg` segments
- Points `orig` and `dest` lay on the semi-major axis of the ellipsoid
- The generated rays sample the volume of the ellipsoid
- Weights are calculated from the Knife-edge diffraction model when parts of the ellipsoid are shadowed
- Initial weights are normalized such that `sum(prod(weights,3),2) = 1`
- Inputs `orig` and `dest` may be provided as double or single precision
- Supported datatypes `dtype` are `float` or `double`

## Declaration:
```
void generate_diffraction_paths(
                const arma::Mat<dtype> *orig, 
                const arma::Mat<dtype> *dest,
                dtype center_frequency, 
                int lod,
                arma::Cube<dtype> *ray_x, 
                arma::Cube<dtype> *ray_y, 
                arma::Cube<dtype> *ray_z,
                arma::Cube<dtype> *weight);
```

## Arguments:
- `const arma::Mat<dtype> ***orig**` (input)<br>
  Pointer to Armadillo matrix containing the origin points of the propagation ellipsoid (e.g.
  transmitter positions). Size: `[ n_pos, 3 ]`

- `const arma::Mat<dtype> ***dest**` (input)<br>
  Pointer to Armadillo matrix containing the destination point of the propagation ellipsoid (e.g.
  receiver positions). Size: `[ n_pos, 3 ]`

- `dtype **center_frequency**` (input)<br>
  The center frequency in [Hz], scalar, default = 299792458 Hz

- `int **lod**` (input)<br>
  Level of detail, scalar value
  `lod = 1` | results in `n_path = 7` and `n_seg = 3`
  `lod = 2` | results in `n_path = 19` and `n_seg = 3`
  `lod = 3` | results in `n_path = 37` and `n_seg = 4`
  `lod = 4` | results in `n_path = 61` and `n_seg = 5`
  `lod = 5` | results in `n_path = 1` and `n_seg = 2` (for debugging)
  `lod = 6` | results in `n_path = 2` and `n_seg = 2` (for debugging)

- `arma::Cube<dtype> ***ray_x**` (output)<br>
  Pointer to an Armadillo cube for the x-coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1 ]`
  Size will be adjusted if not set correctly.

- `arma::Cube<dtype> ***ray_y**` (output)<br>
  Pointer to an Armadillo cube for the y-coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1 ]`
  Size will be adjusted if not set correctly.

- `arma::Cube<dtype> ***ray_z**` (output)<br>
  Pointer to an Armadillo cube for the z-coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1 ]`
  Size will be adjusted if not set correctly.

- `arma::Cube<dtype> ***weight**` (output)<br>
  Pointer to an Armadillo cube for the  weights; Size: `[ n_pos, n_path, n_seg ]`
  Size will be adjusted if not set correctly.

## See also:
- <a href="#calc_diffraction_gain">calc_diffraction_gain</a>
MD!*/

template <typename dtype>
void quadriga_lib::generate_diffraction_paths(const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest, dtype center_frequency, int lod,
                                              arma::Cube<dtype> *ray_x, arma::Cube<dtype> *ray_y, arma::Cube<dtype> *ray_z, arma::Cube<dtype> *weight)
{
    // Check data validity
    if (orig == nullptr || orig->n_rows == 0ULL)
        throw std::invalid_argument("Input 'orig' cannot be empty or NULL.");
    if (dest == nullptr || dest->n_rows == 0ULL)
        throw std::invalid_argument("Input 'dest' cannot be empty or NULL.");
    if (orig->n_cols != 3ULL)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing the x,y,z coordinates.");
    if (dest->n_cols != 3ULL)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing the x,y,z coordinates.");
    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Input 'center_frequency' must be larger that 0.");
    if (ray_x == nullptr || ray_y == nullptr || ray_z == nullptr || weight == nullptr)
        throw std::invalid_argument("Outputs 'ray_x', 'ray_y', 'ray_z' and 'weight' cannot be NULL.");

    arma::uword n_pos = orig->n_rows;
    if (dest->n_rows != n_pos)
        throw std::invalid_argument("Inputs 'orig' and 'dest' must have the same number of rows.");

    arma::uword n_path = 0, n_seg = 0;
    if (lod == 1)
        n_seg = 2, n_path = 7;
    else if (lod == 2)
        n_seg = 2, n_path = 19;
    else if (lod == 3)
        n_seg = 3, n_path = 37;
    else if (lod == 4)
        n_seg = 4, n_path = 61;
    else if (lod == 5)
        n_seg = 1, n_path = 1;
    else if (lod == 6)
        n_seg = 1, n_path = 2;
    else
        throw std::invalid_argument("Input 'lod' must be 1-6.");

    // Adjust output size, if needed
    if (ray_x->n_rows != n_pos || ray_x->n_cols != n_path || ray_x->n_slices != n_seg)
        ray_x->set_size(n_pos, n_path, n_seg);
    if (ray_y->n_rows != n_pos || ray_y->n_cols != n_path || ray_y->n_slices != n_seg)
        ray_y->set_size(n_pos, n_path, n_seg);
    if (ray_z->n_rows != n_pos || ray_z->n_cols != n_path || ray_z->n_slices != n_seg)
        ray_z->set_size(n_pos, n_path, n_seg);
    if (weight->n_rows != n_pos || weight->n_cols != n_path || weight->n_slices != n_seg + 1)
        weight->set_size(n_pos, n_path, n_seg + 1);

    // Normalized ellipsoid coordinates and weights
    arma::vec tx, ty, tz, tw;
    if (lod == 1)
    {
        tx = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
              0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75};
        ty = {0.0, 0.6375, 0.1708, -0.4667, -0.6375, -0.1708, 0.4667,
              0.0, 0.6375, 0.1708, -0.4667, -0.6375, -0.1708, 0.4667};
        tz = {0.0, 0.1708, 0.6375, 0.4667, -0.1708, -0.6375, -0.4667,
              0.0, 0.1708, 0.6375, 0.4667, -0.1708, -0.6375, -0.4667};
        tw = {0.55, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075};
    }
    else if (lod == 2)
    {
        tx = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
              0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75};
        ty = {0.0, 0.7341, 0.1967, -0.5374, -0.7341, -0.1967, 0.5374, 2.1, 1.8187, 1.05, 0.0, -1.05, -1.8187, -2.1, -1.8187, -1.05, 0.0, 1.05, 1.8187,
              0.0, 0.7341, 0.1967, -0.5374, -0.7341, -0.1967, 0.5374, 2.1, 1.8187, 1.05, 0.0, -1.05, -1.8187, -2.1, -1.8187, -1.05, 0.0, 1.05, 1.8187};
        tz = {0.0, 0.1967, 0.7341, 0.5374, -0.1967, -0.7341, -0.5374, 0.0, 1.05, 1.8187, 2.1, 1.8187, 1.05, 0.0, -1.05, -1.8187, -2.1, -1.8187, -1.05,
              0.0, 0.1967, 0.7341, 0.5374, -0.1967, -0.7341, -0.5374, 0.0, 1.05, 1.8187, 2.1, 1.8187, 1.05, 0.0, -1.05, -1.8187, -2.1, -1.8187, -1.05};
        tw = {0.630004, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333};
    }
    else if (lod == 3)
    {
        tx = {0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464,
              0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
              0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536};
        ty = {0.0, 0.316, 0.0847, -0.2314, -0.316, -0.0847, 0.2314, 0.9504, 0.8231, 0.4752, 0.0, -0.4752, -0.8231, -0.9504, -0.8231, -0.4752, 0.0, 0.4752, 0.8231, 2.0854, 1.8658, 1.4211, 0.8049, 0.0917, -0.6325, -1.2805, -1.774, -2.0536, -2.0854, -1.8658, -1.4211, -0.8049, -0.0917, 0.6325, 1.2805, 1.774, 2.0536,
              0.0, 0.4057, 0.1087, -0.297, -0.4057, -0.1087, 0.297, 1.22, 1.0566, 0.61, 0.0, -0.61, -1.0566, -1.22, -1.0566, -0.61, 0.0, 0.61, 1.0566, 2.6769, 2.3949, 1.8241, 1.0332, 0.1178, -0.8119, -1.6437, -2.2772, -2.636, -2.6769, -2.3949, -1.8241, -1.0332, -0.1178, 0.8119, 1.6437, 2.2772, 2.636,
              0.0, 0.316, 0.0847, -0.2314, -0.316, -0.0847, 0.2314, 0.9504, 0.8231, 0.4752, 0.0, -0.4752, -0.8231, -0.9504, -0.8231, -0.4752, 0.0, 0.4752, 0.8231, 2.0854, 1.8658, 1.4211, 0.8049, 0.0917, -0.6325, -1.2805, -1.774, -2.0536, -2.0854, -1.8658, -1.4211, -0.8049, -0.0917, 0.6325, 1.2805, 1.774, 2.0536};
        tz = {0.0, 0.0847, 0.316, 0.2314, -0.0847, -0.316, -0.2314, 0.0, 0.4752, 0.8231, 0.9504, 0.8231, 0.4752, 0.0, -0.4752, -0.8231, -0.9504, -0.8231, -0.4752, 0.2746, 0.9713, 1.5508, 1.9433, 2.1014, 2.0061, 1.6688, 1.1302, 0.4553, -0.2746, -0.9713, -1.5508, -1.9433, -2.1014, -2.0061, -1.6688, -1.1302, -0.4553,
              0.0, 0.1087, 0.4057, 0.297, -0.1087, -0.4057, -0.297, 0.0, 0.61, 1.0566, 1.22, 1.0566, 0.61, 0.0, -0.61, -1.0566, -1.22, -1.0566, -0.61, 0.3524, 1.2467, 1.9906, 2.4945, 2.6974, 2.575, 2.1421, 1.4507, 0.5844, -0.3524, -1.2467, -1.9906, -2.4945, -2.6974, -2.575, -2.1421, -1.4507, -0.5844,
              0.0, 0.0847, 0.316, 0.2314, -0.0847, -0.316, -0.2314, 0.0, 0.4752, 0.8231, 0.9504, 0.8231, 0.4752, 0.0, -0.4752, -0.8231, -0.9504, -0.8231, -0.4752, 0.2746, 0.9713, 1.5508, 1.9433, 2.1014, 2.0061, 1.6688, 1.1302, 0.4553, -0.2746, -0.9713, -1.5508, -1.9433, -2.1014, -2.0061, -1.6688, -1.1302, -0.4553};
        tw = {0.51001, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333};
    }
    else if (lod == 4)
    {
        tx = {0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955,
              0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455,
              0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545,
              0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045};
        ty = {0.0, 0.2077, 0.0556, -0.152, -0.2077, -0.0556, 0.152, 0.6249, 0.5412, 0.3124, 0.0, -0.3124, -0.5412, -0.6249, -0.5412, -0.3124, 0.0, 0.3124, 0.5412, 1.279, 1.1443, 0.8715, 0.4937, 0.0563, -0.3879, -0.7853, -1.088, -1.2595, -1.279, -1.1443, -0.8715, -0.4937, -0.0563, 0.3879, 0.7853, 1.088, 1.2595, 2.5532, 2.4662, 2.2112, 1.8054, 1.2766, 0.6608, 0.0, -0.6608, -1.2766, -1.8054, -2.2112, -2.4662, -2.5532, -2.4662, -2.2112, -1.8054, -1.2766, -0.6608, 0.0, 0.6608, 1.2766, 1.8054, 2.2112, 2.4662,
              0.0, 0.3091, 0.0828, -0.2263, -0.3091, -0.0828, 0.2263, 0.93, 0.8054, 0.465, 0.0, -0.465, -0.8054, -0.93, -0.8054, -0.465, 0.0, 0.465, 0.8054, 1.9035, 1.7031, 1.2971, 0.7347, 0.0837, -0.5773, -1.1688, -1.6193, -1.8745, -1.9035, -1.7031, -1.2971, -0.7347, -0.0837, 0.5773, 1.1688, 1.6193, 1.8745, 3.8, 3.6704, 3.2908, 2.687, 1.9, 0.9835, 0.0, -0.9835, -1.9, -2.687, -3.2908, -3.6704, -3.8, -3.6704, -3.2908, -2.687, -1.9, -0.9835, 0.0, 0.9835, 1.9, 2.687, 3.2908, 3.6704,
              0.0, 0.3091, 0.0828, -0.2263, -0.3091, -0.0828, 0.2263, 0.93, 0.8054, 0.465, 0.0, -0.465, -0.8054, -0.93, -0.8054, -0.465, 0.0, 0.465, 0.8054, 1.9036, 1.7031, 1.2971, 0.7348, 0.0837, -0.5774, -1.1688, -1.6193, -1.8745, -1.9036, -1.7031, -1.2971, -0.7348, -0.0837, 0.5774, 1.1688, 1.6193, 1.8745, 3.8, 3.6705, 3.2909, 2.687, 1.9, 0.9835, 0.0, -0.9835, -1.9, -2.687, -3.2909, -3.6705, -3.8, -3.6705, -3.2909, -2.687, -1.9, -0.9835, 0.0, 0.9835, 1.9, 2.687, 3.2909, 3.6705,
              0.0, 0.2077, 0.0556, -0.152, -0.2077, -0.0556, 0.152, 0.6249, 0.5411, 0.3124, 0.0, -0.3124, -0.5411, -0.6249, -0.5411, -0.3124, 0.0, 0.3124, 0.5411, 1.279, 1.1443, 0.8715, 0.4937, 0.0563, -0.3879, -0.7853, -1.088, -1.2595, -1.279, -1.1443, -0.8715, -0.4937, -0.0563, 0.3879, 0.7853, 1.088, 1.2595, 2.5532, 2.4662, 2.2111, 1.8054, 1.2766, 0.6608, 0.0, -0.6608, -1.2766, -1.8054, -2.2111, -2.4662, -2.5532, -2.4662, -2.2111, -1.8054, -1.2766, -0.6608, 0.0, 0.6608, 1.2766, 1.8054, 2.2111, 2.4662};
        tz = {0.0, 0.0556, 0.2077, 0.152, -0.0556, -0.2077, -0.152, 0.0, 0.3124, 0.5412, 0.6249, 0.5412, 0.3124, 0.0, -0.3124, -0.5412, -0.6249, -0.5412, -0.3124, 0.1684, 0.5957, 0.9511, 1.1918, 1.2888, 1.2303, 1.0235, 0.6931, 0.2792, -0.1684, -0.5957, -0.9511, -1.1918, -1.2888, -1.2303, -1.0235, -0.6931, -0.2792, 0.0, 0.6608, 1.2766, 1.8054, 2.2112, 2.4662, 2.5532, 2.4662, 2.2112, 1.8054, 1.2766, 0.6608, 0.0, -0.6608, -1.2766, -1.8054, -2.2112, -2.4662, -2.5532, -2.4662, -2.2112, -1.8054, -1.2766, -0.6608,
              0.0, 0.0828, 0.3091, 0.2263, -0.0828, -0.3091, -0.2263, 0.0, 0.465, 0.8054, 0.93, 0.8054, 0.465, 0.0, -0.465, -0.8054, -0.93, -0.8054, -0.465, 0.2506, 0.8865, 1.4155, 1.7738, 1.9181, 1.8311, 1.5232, 1.0316, 0.4156, -0.2506, -0.8865, -1.4155, -1.7738, -1.9181, -1.8311, -1.5232, -1.0316, -0.4156, 0.0, 0.9835, 1.9, 2.687, 3.2908, 3.6704, 3.8, 3.6704, 3.2908, 2.687, 1.9, 0.9835, 0.0, -0.9835, -1.9, -2.687, -3.2908, -3.6704, -3.8, -3.6704, -3.2908, -2.687, -1.9, -0.9835,
              0.0, 0.0828, 0.3091, 0.2263, -0.0828, -0.3091, -0.2263, 0.0, 0.465, 0.8054, 0.93, 0.8054, 0.465, 0.0, -0.465, -0.8054, -0.93, -0.8054, -0.465, 0.2506, 0.8866, 1.4156, 1.7738, 1.9182, 1.8311, 1.5232, 1.0316, 0.4156, -0.2506, -0.8866, -1.4156, -1.7738, -1.9182, -1.8311, -1.5232, -1.0316, -0.4156, 0.0, 0.9835, 1.9, 2.687, 3.2909, 3.6705, 3.8, 3.6705, 3.2909, 2.687, 1.9, 0.9835, 0.0, -0.9835, -1.9, -2.687, -3.2909, -3.6705, -3.8, -3.6705, -3.2909, -2.687, -1.9, -0.9835,
              0.0, 0.0556, 0.2077, 0.152, -0.0556, -0.2077, -0.152, 0.0, 0.3124, 0.5411, 0.6249, 0.5411, 0.3124, 0.0, -0.3124, -0.5411, -0.6249, -0.5411, -0.3124, 0.1684, 0.5957, 0.9511, 1.1918, 1.2888, 1.2303, 1.0235, 0.6931, 0.2792, -0.1684, -0.5957, -0.9511, -1.1918, -1.2888, -1.2303, -1.0235, -0.6931, -0.2792, 0.0, 0.6608, 1.2766, 1.8054, 2.2111, 2.4662, 2.5532, 2.4662, 2.2111, 1.8054, 1.2766, 0.6608, 0.0, -0.6608, -1.2766, -1.8054, -2.2111, -2.4662, -2.5532, -2.4662, -2.2111, -1.8054, -1.2766, -0.6608};
        tw = {0.490012, 0.043333, 0.043333, 0.043333, 0.043333, 0.043333, 0.043333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125};
    }
    else if (lod == 5)
    {
        tx = {0.5};
        ty = {0.0};
        tz = {0.0};
        tw = {1.0};
    }
    else if (lod == 6)
    {
        tx = {0.5, 0.5};
        ty = {0.0, 0.0};
        tz = {0.0, 0.0};
        tw = {0.5, 0.5};
    }
    else
        throw std::invalid_argument("Input 'lod' must be 1-6.");

    // Copy weights to output memory
    double *p_tw = tw.memptr();
    double s = 1.0 / double(n_seg + 1);
    for (arma::uword i = 0; i < n_path; ++i)
        p_tw[i] = std::pow(p_tw[i], s);
    qd_repeat_sequence(p_tw, n_path, n_pos, n_seg + 1, weight->memptr());

    // Pointers
    auto p_ox = orig->colptr(0), p_oy = orig->colptr(1), p_oz = orig->colptr(2); // Origin pointer (dtype)
    auto p_dx = dest->colptr(0), p_dy = dest->colptr(1), p_dz = dest->colptr(2); // Destination pointer (dtype)
    auto p_tx = tx.memptr(), p_ty = ty.memptr(), p_tz = tz.memptr();             // Path pointer (double)
    auto p_rx = ray_x->memptr(), p_ry = ray_y->memptr(), p_rz = ray_z->memptr(); // Ray pointer (dtype)

    // Constants
    double lambda_div_8 = 0.125 * 299792458.0 / (double)center_frequency; // lambda / 8

    // Iterate through positions
    size_t n_pos_t = (size_t)n_pos;
    size_t n_ray_t = size_t(n_path * n_seg);
    for (size_t i = 0; i < n_pos_t; ++i)
    {
        // Calculate ellipsoid orientation and length of the semi-major axis
        double ox = (double)p_ox[i], oy = (double)p_oy[i], oz = (double)p_oz[i];
        double x = (double)p_dx[i] - ox,
               y = (double)p_dy[i] - oy,
               z = (double)p_dz[i] - oz;
        double d3d = std::sqrt(x * x + y * y + z * z), r_d3d = 1.0 / d3d;
        x *= r_d3d, y *= r_d3d, z *= r_d3d;
        z = (z > 1.0) ? 1.0 : (z < -1.0 ? -1.0 : z);
        double az = std::atan2(y, x);
        double el = std::asin(z);

        double sin_el = z, cos_el = std::cos(el);
        double sin_az = std::sin(az), cos_az = std::cos(az);

        // Calculate ray coordinates
        double width = std::sqrt(d3d * lambda_div_8);
        for (size_t j = 0; j < n_ray_t; ++j)
        {
            // Read normalized coordinates
            x = p_tx[j], y = p_ty[j], z = p_tz[j];

            // Scale length and width of the ellipsoid
            x *= d3d, y *= width, z *= width;

            // Rotate the ellipsoid
            double tmp = cos_el * x - sin_el * z;
            z = sin_el * x + cos_el * z, x = tmp;
            tmp = cos_az * x - sin_az * y;
            y = sin_az * x + cos_az * y, x = tmp;

            // Add origin
            x += ox, y += oy, z += oz;

            // Convert type and write to output
            size_t ij = j * n_pos_t + i;
            p_rx[ij] = (dtype)x;
            p_ry[ij] = (dtype)y;
            p_rz[ij] = (dtype)z;
        }
    }
}
template void quadriga_lib::generate_diffraction_paths(const arma::Mat<float> *orig, const arma::Mat<float> *dest, float center_frequency, int lod,
                                                       arma::Cube<float> *ray_x, arma::Cube<float> *ray_y, arma::Cube<float> *ray_z, arma::Cube<float> *weight);

template void quadriga_lib::generate_diffraction_paths(const arma::Mat<double> *orig, const arma::Mat<double> *dest, double center_frequency, int lod,
                                                       arma::Cube<double> *ray_x, arma::Cube<double> *ray_y, arma::Cube<double> *ray_z, arma::Cube<double> *weight);
