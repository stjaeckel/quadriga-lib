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

#include "quadriga_tools.hpp"
#include <vector>
#include <fstream>
#include <stdexcept>

/*!SECTION
Miscellaneous / Tools
SECTION!*/

// Helper function : repeat sequence of values
template <typename dtypeIn, typename dtypeOut>
static void qd_repeat_sequence(const dtypeIn *sequence, arma::uword sequence_length, arma::uword repeat_value, arma::uword repeat_sequence, dtypeOut *output)
{
    arma::uword pos = 0;                                  // Position in output
    for (arma::uword rs = 0; rs < repeat_sequence; ++rs)  // Repeat sequence of values
        for (arma::uword v = 0; v < sequence_length; ++v) // Iterate through all values of the sequence
        {
            dtypeOut val = (dtypeOut)sequence[v];             // Type conversion
            for (arma::uword rv = 0; rv < repeat_value; ++rv) // Repeat each value
                output[pos++] = val;
        }
}

// FUNCTION: Calculate rotation matrix R from roll, pitch, and yaw angles (given by rows in the input "orientation")
template <typename dtype>
arma::cube quadriga_lib::calc_rotation_matrix(const arma::Cube<dtype> orientation, bool invert_y_axis, bool transposeR)
{
    // Input:       orientation         Orientation vectors (rows = bank (roll), tilt (pitch), heading (yaw)) in [rad], Size [3, n_row, n_col]
    //              invert_y_axis       Inverts the y-axis
    //              transposeR          Returns the transpose of R instead of R
    // Output:      R                   Rotation matrix, column-major order, Size [9, n_row, n_col ]

    if (orientation.n_elem == 0)
        throw std::invalid_argument("Input cannot be empty.");
    if (orientation.n_rows != 3)
        throw std::invalid_argument("Input must have 3 rows.");

    unsigned long long n_row = orientation.n_cols, n_col = orientation.n_slices;
    arma::cube rotation = arma::cube(9, n_row, n_col, arma::fill::zeros); // Always double precision
    const dtype *p_orientation = orientation.memptr();
    double *p_rotation = rotation.memptr();

    for (auto iC = 0ULL; iC < n_col; ++iC)
        for (auto iR = 0ULL; iR < n_row; ++iR)
        {
            double cc = (double)*p_orientation++, sc = sin(cc);
            double cb = (double)*p_orientation++, sb = sin(cb);
            double ca = (double)*p_orientation++, sa = sin(ca);
            ca = cos(ca), cb = cos(cb), cc = cos(cc), sb = invert_y_axis ? -sb : sb;

            if (transposeR)
            {
                *p_rotation++ = ca * cb;
                *p_rotation++ = ca * sb * sc - sa * cc;
                *p_rotation++ = ca * sb * cc + sa * sc;
                *p_rotation++ = sa * cb;
                *p_rotation++ = sa * sb * sc + ca * cc;
                *p_rotation++ = sa * sb * cc - ca * sc;
                *p_rotation++ = -sb;
                *p_rotation++ = cb * sc;
                *p_rotation++ = cb * cc;
            }
            else
            {
                *p_rotation++ = ca * cb;
                *p_rotation++ = sa * cb;
                *p_rotation++ = -sb;
                *p_rotation++ = ca * sb * sc - sa * cc;
                *p_rotation++ = sa * sb * sc + ca * cc;
                *p_rotation++ = cb * sc;
                *p_rotation++ = ca * sb * cc + sa * sc;
                *p_rotation++ = sa * sb * cc - ca * sc;
                *p_rotation++ = cb * cc;
            }
        }

    return rotation;
}
template arma::cube quadriga_lib::calc_rotation_matrix(const arma::Cube<float>, bool invert_y_axis, bool transposeR);
template arma::cube quadriga_lib::calc_rotation_matrix(const arma::Cube<double>, bool invert_y_axis, bool transposeR);

/*!MD
# generate_diffraction_paths
Generate propagation paths for estimating the diffraction gain

## Description:
Diffraction refers to the phenomenon where waves bend or interfere around the edges of an obstacle,
extending into the region that would otherwise be in the obstacle's geometrical shadow. The object
causing the diffraction acts as a secondary source for the wave's propagation. A specific example of
this is the knife-edge effect, or knife-edge diffraction, where a sharp, well-defined obstacle—like
a mountain range or a building wall—partially truncates the incident radiation.<br><br>

To estimate the diffraction gain in a three-dimensional space, one can assess the extent to which the
Fresnel ellipsoid is obstructed by objects, and then evaluate the impact of this obstruction on the
received power. This method presupposes that diffracted waves travel along slightly varied paths
before arriving at a receiver. These waves may reach the receiver out of phase with the primary wave
due to their different travel lengths, leading to either constructive or destructive interference.<br><br>

The process of estimating the gain involves dividing the wave propagation from a transmitter to a
receiver into `n_path` paths. These paths are represented by elliptic arcs, which are further
approximated using `n_seg` line segments. Each segment can be individually blocked or attenuated
by environmental objects. To determine the overall diffraction gain, a weighted sum of these
individual path contributions is calculated. The weighting is adjusted to align with the uniform
theory of diffraction (UTD) coefficients in two dimensions, but the methodology is adapted for
any 3D object shape. This function generates the elliptic propagation paths and corresponding weights
necessary for this calculation.

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
void generate_diffraction_paths(const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest,
                                dtype center_frequency, int lod,
                                arma::Cube<dtype> *ray_x, arma::Cube<dtype> *ray_y,
                                arma::Cube<dtype> *ray_z, arma::Cube<dtype> *weight);
```

## Arguments:
- **`const arma::Mat<dtype> *orig`**<br>
  Pointer to Armadillo matrix containing the origin points of the propagation ellipsoid (e.g.
  transmitter positions). Size: `[ n_pos, 3 ]`

- **`const arma::Mat<dtype> *dest`**<br>
  Pointer to Armadillo matrix containing the destination point of the propagation ellipsoid (e.g.
  receiver positions). Size: `[ n_pos, 3 ]`

- **`dtype center_frequency`**<br>
  The center frequency in [Hz], scalar, default = 299792458 Hz

- **`int lod`**<br>
  Level of detail, scalar value
  `lod = 1` | results in `n_path = 7` and `n_seg = 3`
  `lod = 2` | results in `n_path = 19` and `n_seg = 3`
  `lod = 3` | results in `n_path = 37` and `n_seg = 4`
  `lod = 4` | results in `n_path = 61` and `n_seg = 5`
  `lod = 5` | results in `n_path = 1` and `n_seg = 2` (for debugging)
  `lod = 6` | results in `n_path = 2` and `n_seg = 2` (for debugging)

- **`arma::Cube<dtype> *ray_x`**<br>
  Pointer to an Armadillo cube for the x-coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1 ]`
  Size will be adjusted if not set correctly.

- **`arma::Cube<dtype> *ray_y`**<br>
  Pointer to an Armadillo cube for the y-coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1 ]`
  Size will be adjusted if not set correctly.

- **`arma::Cube<dtype> *ray_z`**<br>
  Pointer to an Armadillo cube for the z-coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1 ]`
  Size will be adjusted if not set correctly.

- **`arma::Cube<dtype> *weight`**<br>
  Pointer to an Armadillo cube for the  weights; Size: `[ n_pos, n_path, n_seg ]`
  Size will be adjusted if not set correctly.
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

// FUNCTION: Transform from geographic coordinates to Cartesian coordinates
template <typename dtype>
arma::cube quadriga_lib::geo2cart(const arma::Mat<dtype> azimuth, const arma::Mat<dtype> elevation, const arma::Mat<dtype> length)
{
    // Inputs:          azimuth         Azimuth angles in [rad],                Size [n_row, n_col]
    //                  elevation       Elevation angles in [rad],              Size [n_row, n_col]
    //                  length          Length of the vector,                   Size [n_row, n_col]
    // Output:          cart            Cartesian coordinates,                  Size [3, n_row, n_col]

    if (azimuth.n_elem == 0 || elevation.n_elem == 0 || length.n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (elevation.n_rows != azimuth.n_rows || length.n_rows != azimuth.n_rows ||
        elevation.n_cols != azimuth.n_cols || length.n_cols != azimuth.n_cols)
        throw std::invalid_argument("Inputs must have the same size.");

    unsigned long long n_row = azimuth.n_rows, n_col = azimuth.n_cols;
    arma::cube cart = arma::cube(3, n_row, n_col, arma::fill::zeros); // Always double precision

    for (auto i = 0ULL; i < azimuth.n_elem; ++i)
    {
        double ca = (double)azimuth(i), sa = sin(ca), r = (double)length(i);
        double ce = (double)elevation(i), se = sin(ce);
        ca = cos(ca), ce = cos(ce);

        unsigned long long rw = i % n_row, co = i / n_row;
        cart(0, rw, co) = r * ce * ca;
        cart(1, rw, co) = r * ce * sa;
        cart(2, rw, co) = r * se;
    }
    return cart;
}
template arma::cube quadriga_lib::geo2cart(const arma::Mat<float> azimuth, const arma::Mat<float> elevation, const arma::Mat<float> length);
template arma::cube quadriga_lib::geo2cart(const arma::Mat<double> azimuth, const arma::Mat<double> elevation, const arma::Mat<double> length);

// FUNCTION: Transform from Cartesian coordinates to geographic coordinates
template <typename dtype>
arma::cube quadriga_lib::cart2geo(const arma::Cube<dtype> cart)
{
    // Input:           cart            Cartesian coordinates,                  Size [3, n_row, n_col]
    // Output:          geo             geographic coordinates (az,el,len)      Size [n_row, n_col, 3]

    if (cart.n_elem == 0)
        throw std::invalid_argument("Input cannot be empty.");
    if (cart.n_rows != 3)
        throw std::invalid_argument("Input must have 3 rows.");

    unsigned long long n_row = cart.n_cols, n_col = cart.n_slices;
    arma::cube geo = arma::cube(n_row, n_col, 3, arma::fill::zeros); // Always double precision

    for (auto r = 0ULL; r < n_row; ++r)
        for (auto c = 0ULL; c < n_col; ++c)
        {
            double x = (double)cart(0, r, c), y = (double)cart(1, r, c), z = (double)cart(2, r, c);
            double len = sqrt(x * x + y * y + z * z), rlen = 1.0 / len;
            x *= rlen, y *= rlen, z *= rlen;
            x = x > 1.0 ? 1.0 : x, y = y > 1.0 ? 1.0 : y, z = z > 1.0 ? 1.0 : z;
            geo(r, c, 0) = atan2(y, x);
            geo(r, c, 1) = asin(z);
            geo(r, c, 2) = len;
        }
    return geo;
}
template arma::cube quadriga_lib::cart2geo(const arma::Cube<float> cart);
template arma::cube quadriga_lib::cart2geo(const arma::Cube<double> cart);

// Convert path interaction coordinates into FBS/LBS positions, path length and angles
template <typename dtype>
void quadriga_lib::coord2path(dtype Tx, dtype Ty, dtype Tz, dtype Rx, dtype Ry, dtype Rz,
                              const arma::Col<unsigned> *no_interact, const arma::Mat<dtype> *interact_coord,
                              arma::Col<dtype> *path_length, arma::Mat<dtype> *fbs_pos, arma::Mat<dtype> *lbs_pos,
                              arma::Mat<dtype> *path_angles)
{
    if (no_interact == nullptr)
        throw std::invalid_argument("Input 'no_interact' cannot be NULL.");

    unsigned long long n_path = no_interact->n_elem;

    // Calculate the total number of interactions
    unsigned interact_cnt = 0;
    const unsigned *p_interact = no_interact->memptr();
    for (auto i = 0ULL; i < n_path; ++i)
        interact_cnt += p_interact[i];

    unsigned long long n_interact = (unsigned long long)interact_cnt;

    if (interact_coord == nullptr || interact_coord->n_rows != 3ULL)
        throw std::invalid_argument("Input 'interact_coord' must have 3 rows.");

    if (interact_coord->n_cols != n_interact)
        throw std::invalid_argument("Number of columns of 'interact_coord' must match the sum of 'no_interact'.");

    constexpr dtype zero = dtype(0.0);
    constexpr dtype half = dtype(0.5);
    constexpr dtype los_limit = dtype(1.0e-4);

    // Set the output size
    if (path_length != nullptr && path_length->n_elem != n_path)
        path_length->set_size(n_path);
    if (fbs_pos != nullptr && (fbs_pos->n_rows != 3ULL || fbs_pos->n_cols != n_path))
        fbs_pos->set_size(3, n_path);
    if (lbs_pos != nullptr && (lbs_pos->n_rows != 3ULL || lbs_pos->n_cols != n_path))
        lbs_pos->set_size(3, n_path);
    if (path_angles != nullptr && (path_angles->n_rows != n_path || path_angles->n_cols != 4ULL))
        path_angles->set_size(n_path, 4ULL);

    // Get pointers
    const dtype *p_coord = interact_coord->memptr();
    dtype *p_length = path_length == nullptr ? nullptr : path_length->memptr();
    dtype *p_fbs = fbs_pos == nullptr ? nullptr : fbs_pos->memptr();
    dtype *p_lbs = lbs_pos == nullptr ? nullptr : lbs_pos->memptr();
    dtype *p_angles = path_angles == nullptr ? nullptr : path_angles->memptr();

    // Calculate half way point between TX and RX
    dtype TRx = Rx - Tx, TRy = Ry - Ty, TRz = Rz - Tz;
    TRx = Tx + half * TRx, TRy = Ty + half * TRy, TRz = Tz + half * TRz;

    for (auto ip = 0ULL; ip < n_path; ++ip)
    {
        dtype fx = TRx, fy = TRy, fz = TRz;     // Initial FBS-Pos = half way point
        dtype lx = TRx, ly = TRy, lz = TRz;     // Initial LBS-Pos = half way point
        dtype x = Tx, y = Ty, z = Tz, d = zero; // Set segment start to TX position

        // Get FBS and LBS positions
        for (unsigned ii = 0; ii < p_interact[ip]; ++ii)
        {
            lx = *p_coord++, ly = *p_coord++, lz = *p_coord++;                      // Read segment end coordinate
            x -= lx, y -= ly, z -= lz;                                              // Calculate vector pointing from segment start to segment end
            d += std::sqrt(x * x + y * y + z * z);                                  // Add segment length to total path length
            x = lx, y = ly, z = lz;                                                 // Update segment start for next segment
            fx = ii == 0 ? lx : fx, fy = ii == 0 ? ly : fy, fz = ii == 0 ? lz : fz; // Sore FBS position (segment 0)
        }
        x -= Rx, y -= Ry, z -= Rz;             // Calculate vector pointing last segment start to RX position
        d += std::sqrt(x * x + y * y + z * z); // Add last segment length to total path length

        if (p_length != nullptr)
            p_length[ip] = d;
        if (p_fbs != nullptr)
            p_fbs[3 * ip] = fx, p_fbs[3 * ip + 1] = fy, p_fbs[3 * ip + 2] = fz;
        if (p_lbs != nullptr)
            p_lbs[3 * ip] = lx, p_lbs[3 * ip + 1] = ly, p_lbs[3 * ip + 2] = lz;

        if (p_angles != nullptr)
        {
            x = fx - Tx, y = fy - Ty, z = fz - Tz;
            d = std::sqrt(x * x + y * y + z * z);
            p_angles[ip] = std::atan2(y, x);                                 // AOD
            p_angles[n_path + ip] = d < los_limit ? zero : std::asin(z / d); // EOD
            x = lx - Rx, y = ly - Ry, z = lz - Rz;
            d = std::sqrt(x * x + y * y + z * z);
            p_angles[2 * n_path + ip] = std::atan2(y, x);                        // AOA
            p_angles[3 * n_path + ip] = d < los_limit ? zero : std::asin(z / d); // EOA
        }
    }
}
template void quadriga_lib::coord2path(float Tx, float Ty, float Tz, float Rx, float Ry, float Rz, const arma::Col<unsigned> *no_interact, const arma::Mat<float> *interact_coord,
                                       arma::Col<float> *path_length, arma::Mat<float> *fbs_pos, arma::Mat<float> *lbs_pos, arma::Mat<float> *path_angles);
template void quadriga_lib::coord2path(double Tx, double Ty, double Tz, double Rx, double Ry, double Rz, const arma::Col<unsigned> *no_interact, const arma::Mat<double> *interact_coord,
                                       arma::Col<double> *path_length, arma::Mat<double> *fbs_pos, arma::Mat<double> *lbs_pos, arma::Mat<double> *path_angles);

// Construct a geodesic polyhedron (icosphere), a convex polyhedron made from triangles
template <typename dtype>
unsigned long long quadriga_lib::icosphere(unsigned long long n_div, dtype radius, arma::Mat<dtype> *center, arma::Col<dtype> *length, arma::Mat<dtype> *vert, arma::Mat<dtype> *direction)
{
    if (n_div == 0ULL)
        throw std::invalid_argument("Input 'n_div' cannot be 0.");

    if (radius < dtype(0.0))
        throw std::invalid_argument("Input 'radius' cannot be negative.");

    unsigned long long n_faces = n_div * n_div * 20ULL;

    // Set sizes
    if (center == nullptr)
        throw std::invalid_argument("Output 'center' cannot be NULL.");

    if (center->n_rows != n_faces || center->n_cols != 3ULL)
        center->set_size(n_faces, 3ULL);

    if (length != nullptr && length->n_elem != n_faces)
        length->set_size(n_faces);

    if (vert != nullptr && (vert->n_rows != n_faces || vert->n_cols != 9ULL))
        vert->set_size(n_faces, 9ULL);

    bool calc_directions = direction != nullptr;
    if (calc_directions && (direction->n_rows != n_faces || direction->n_cols != 6ULL))
        direction->set_size(n_faces, 6ULL);

    // Vertex coordinates of a regular isohedron
    dtype r = radius, p = dtype(1.6180340) * r, z = dtype(0.0);
    dtype val[180] = {z, z, z, z, z, z, z, z, -p, -p, p, p, -r, -r, r, z, z, z, z, z,
                      r, r, r, r, r, -r, -r, -r, z, z, z, z, -p, -p, -p, r, r, r, r, r,
                      p, p, p, p, p, p, p, p, r, r, r, r, z, z, z, -p, -p, -p, -p, -p,
                      z, -p, p, -r, r, r, -r, -p, -r, -p, r, p, -p, z, z, -r, -p, z, p, r,
                      -r, z, z, p, p, -p, -p, z, p, z, -p, z, z, -r, -r, p, z, -r, z, p,
                      p, r, r, z, z, z, z, r, z, -r, z, -r, -r, -p, -p, z, -r, -p, -r, z,
                      p, z, r, -p, -r, p, r, -r, -p, -r, p, r, z, r, p, r, -r, -p, z, p,
                      z, -r, p, z, p, z, -p, -p, z, -p, z, p, -r, -p, z, p, p, z, -r, z,
                      r, p, z, r, z, r, z, z, -r, z, -r, z, -p, z, -r, z, z, -r, -p, -r};

    // Rotate x and y-coordinates slightly to avoid artifacts in regular grids
    constexpr dtype si = dtype(0.0078329); // ~ 0.45 degree
    constexpr dtype co = dtype(0.999969322368236);
    for (unsigned n = 0; n < 20; ++n)
    {
        dtype tmp = val[n];
        val[n] = co * tmp - si * val[n + 20];
        val[n + 20] = si * tmp + co * val[n + 20];
        tmp = val[n + 60];
        val[n + 60] = co * tmp - si * val[n + 80];
        val[n + 80] = si * tmp + co * val[n + 80];
        tmp = val[n + 120];
        val[n + 120] = co * tmp - si * val[n + 140];
        val[n + 140] = si * tmp + co * val[n + 140];
    }

    // Convert to armadillo matrix
    arma::Mat<dtype> isohedron = arma::Mat<dtype>(val, 20ULL, 9ULL, false, true);

    // Subdivide faces of the isohedron
    arma::Mat<dtype> icosphere;
    quadriga_lib::subdivide_triangles(n_div, &isohedron, &icosphere);

    // Get pointers for direct memory access
    dtype *p_icosphere = icosphere.memptr();
    dtype *p_dest = center->memptr();
    dtype *p_length = (length == nullptr) ? nullptr : length->memptr();
    dtype *p_trivec = (vert == nullptr) ? nullptr : vert->memptr();
    dtype *p_tridir = calc_directions ? direction->memptr() : nullptr;

    // Process all faces
    constexpr dtype one = dtype(1.0), none = -one;
    dtype ri = one / r;
    for (auto n = 0ULL; n < n_faces; ++n)
    {
        // Project triangles onto the unit sphere
        // First vertex
        dtype tmp = r / std::sqrt(p_icosphere[n] * p_icosphere[n] +
                                  p_icosphere[n + n_faces] * p_icosphere[n + n_faces] +
                                  p_icosphere[n + 2ULL * n_faces] * p_icosphere[n + 2ULL * n_faces]);

        p_icosphere[n] *= tmp;
        p_icosphere[n + n_faces] *= tmp;
        p_icosphere[n + 2ULL * n_faces] *= tmp;

        if (calc_directions)
        {
            dtype tmp = p_icosphere[n + 2ULL * n_faces] * ri;
            tmp = (tmp > one) ? one : (tmp < none ? none : tmp);
            p_tridir[n] = std::atan2(p_icosphere[n + n_faces], p_icosphere[n]);
            p_tridir[n + n_faces] = std::asin(tmp);
        }

        // Second vertex
        tmp = r / std::sqrt(p_icosphere[n + 3ULL * n_faces] * p_icosphere[n + 3ULL * n_faces] +
                            p_icosphere[n + 4ULL * n_faces] * p_icosphere[n + 4ULL * n_faces] +
                            p_icosphere[n + 5ULL * n_faces] * p_icosphere[n + 5ULL * n_faces]);

        p_icosphere[n + 3ULL * n_faces] *= tmp;
        p_icosphere[n + 4ULL * n_faces] *= tmp;
        p_icosphere[n + 5ULL * n_faces] *= tmp;

        if (calc_directions)
        {
            dtype tmp = p_icosphere[n + 5ULL * n_faces] * ri;
            tmp = (tmp > one) ? one : (tmp < none ? none : tmp);
            p_tridir[n + 2ULL * n_faces] = std::atan2(p_icosphere[n + 4ULL * n_faces], p_icosphere[n + 3ULL * n_faces]);
            p_tridir[n + 3ULL * n_faces] = std::asin(tmp);
        }

        // Third vertex
        tmp = r / std::sqrt(p_icosphere[n + 6ULL * n_faces] * p_icosphere[n + 6ULL * n_faces] +
                            p_icosphere[n + 7ULL * n_faces] * p_icosphere[n + 7ULL * n_faces] +
                            p_icosphere[n + 8ULL * n_faces] * p_icosphere[n + 8ULL * n_faces]);

        p_icosphere[n + 6ULL * n_faces] *= tmp;
        p_icosphere[n + 7ULL * n_faces] *= tmp;
        p_icosphere[n + 8ULL * n_faces] *= tmp;

        if (calc_directions)
        {
            dtype tmp = p_icosphere[n + 8ULL * n_faces] * ri;
            tmp = (tmp > one) ? one : (tmp < none ? none : tmp);
            p_tridir[n + 4ULL * n_faces] = std::atan2(p_icosphere[n + 7ULL * n_faces], p_icosphere[n + 6ULL * n_faces]);
            p_tridir[n + 5ULL * n_faces] = std::asin(tmp);
        }

        // Calculate normal vector of the plane that is formed by the 3 vertices
        dtype Ux = p_icosphere[n + 3ULL * n_faces] - p_icosphere[n],
              Uy = p_icosphere[n + 4ULL * n_faces] - p_icosphere[n + n_faces],
              Uz = p_icosphere[n + 5ULL * n_faces] - p_icosphere[n + 2ULL * n_faces];

        dtype Vx = p_icosphere[n + 6ULL * n_faces] - p_icosphere[n],
              Vy = p_icosphere[n + 7ULL * n_faces] - p_icosphere[n + n_faces],
              Vz = p_icosphere[n + 8ULL * n_faces] - p_icosphere[n + 2ULL * n_faces];

        dtype Nx = Uy * Vz - Uz * Vy, Ny = Uz * Vx - Ux * Vz, Nz = Ux * Vy - Uy * Vx;        // Cross Product
        tmp = one / std::sqrt(Nx * Nx + Ny * Ny + Nz * Nz), Nx *= tmp, Ny *= tmp, Nz *= tmp; // Normalize

        // Distance from origin to plane
        tmp = (p_icosphere[n] * Nx + p_icosphere[n + n_faces] * Ny + p_icosphere[n + 2ULL * n_faces] * Nz);

        // Calculate intersect coordinate
        p_dest[n] = tmp * Nx;
        p_dest[n + n_faces] = tmp * Ny;
        p_dest[n + 2ULL * n_faces] = tmp * Nz;

        if (p_length != nullptr)
            p_length[n] = std::abs(tmp);

        // Calculate vectors pointing from the face center to the triangle vertices
        if (p_trivec != nullptr)
        {
            p_trivec[n] = p_icosphere[n] - p_dest[n];
            p_trivec[n + n_faces] = p_icosphere[n + n_faces] - p_dest[n + n_faces];
            p_trivec[n + 2ULL * n_faces] = p_icosphere[n + 2ULL * n_faces] - p_dest[n + 2ULL * n_faces];
            p_trivec[n + 3ULL * n_faces] = p_icosphere[n + 3ULL * n_faces] - p_dest[n];
            p_trivec[n + 4ULL * n_faces] = p_icosphere[n + 4ULL * n_faces] - p_dest[n + n_faces];
            p_trivec[n + 5ULL * n_faces] = p_icosphere[n + 5ULL * n_faces] - p_dest[n + 2ULL * n_faces];
            p_trivec[n + 6ULL * n_faces] = p_icosphere[n + 6ULL * n_faces] - p_dest[n];
            p_trivec[n + 7ULL * n_faces] = p_icosphere[n + 7ULL * n_faces] - p_dest[n + n_faces];
            p_trivec[n + 8ULL * n_faces] = p_icosphere[n + 8ULL * n_faces] - p_dest[n + 2ULL * n_faces];
        }
    }

    return n_faces;
}

template unsigned long long quadriga_lib::icosphere(unsigned long long n_div, float radius, arma::Mat<float> *center, arma::Col<float> *length, arma::Mat<float> *vert, arma::Mat<float> *direction);
template unsigned long long quadriga_lib::icosphere(unsigned long long n_div, double radius, arma::Mat<double> *center, arma::Col<double> *length, arma::Mat<double> *vert, arma::Mat<double> *direction);

// 2D linear interpolation
template <typename dtype>
std::string quadriga_lib::interp(const arma::Cube<dtype> *input, const arma::Col<dtype> *xi, const arma::Col<dtype> *yi,
                                 const arma::Col<dtype> *xo, const arma::Col<dtype> *yo, arma::Cube<dtype> *output)
{
    if (input == nullptr || xi == nullptr || yi == nullptr || xo == nullptr || output == nullptr)
        return "Arguments cannot be NULL.";

    constexpr dtype one = dtype(1.0), zero = dtype(0.0);
    const unsigned long long nx = input->n_cols, ny = input->n_rows, ne = input->n_slices, nxy = nx * ny;
    const unsigned long long mx = xo->n_elem, my = yo->n_elem;

    if (input->n_elem == 0 || xi->n_elem != nx || yi->n_elem != ny || ne == 0)
        return "Data dimensions must match the given number of sample points.";

    if (mx == 0 || my == 0)
        return "Output must have at least one sample point.";

    if (output->n_rows != my || output->n_cols != mx || output->n_slices != ne)
        output->set_size(my, mx, ne);

    unsigned long long *i_xp_vec = new unsigned long long[mx], *i_xn_vec = new unsigned long long[mx];
    dtype *xp_vec = new dtype[mx], *xn_vec = new dtype[mx];

    unsigned long long *i_yp_vec = new unsigned long long[my], *i_yn_vec = new unsigned long long[my];
    dtype *yp_vec = new dtype[my], *yn_vec = new dtype[my];

    {
        // Calculate the x-interpolation parameters
        bool sorted = xi->is_sorted();
        arma::uvec ind = sorted ? arma::regspace<arma::uvec>(0, nx - 1) : arma::sort_index(*xi);
        unsigned long long *p_ind = ind.memptr();
        const dtype *pi = xi->memptr();

        dtype *p_grid_srt = new dtype[nx];
        if (!sorted)
            for (auto i = 0ULL; i < nx; ++i)
                p_grid_srt[i] = pi[p_ind[i]];
        const dtype *p_grid = sorted ? pi : p_grid_srt;

        dtype *p_diff = new dtype[nx];
        *p_diff = one;
        for (auto i = 1ULL; i < nx; ++i)
            p_diff[i] = one / (p_grid[i] - p_grid[i - 1]);

        const dtype *po = xo->memptr();
        for (auto i = 0ULL; i < mx; ++i)
        {
            unsigned long long ip = 0ULL, in = 0ULL; // Indices for reading the input data
            dtype val = po[i], wp = one, wn = zero;  // Relative weights for interpolation
            while (ip < nx && p_grid[ip] <= val)
                ip++;
            if (ip == nx)
                in = --ip;
            else if (ip != 0)
            {
                in = ip--;
                wp = (p_grid[in] - val) * p_diff[in];
                wp = wp > one ? one : wp;
                wp = wp < zero ? zero : wp;
                wn = one - wp;
            }
            i_xp_vec[i] = p_ind[ip], i_xn_vec[i] = p_ind[in], xp_vec[i] = wp, xn_vec[i] = wn;
        }
        delete[] p_diff;
        delete[] p_grid_srt;
        ind.reset();
    }
    {
        // Calculate the y-interpolation parameters
        bool sorted = yi->is_sorted();
        arma::uvec ind = sorted ? arma::regspace<arma::uvec>(0, ny - 1) : arma::sort_index(*yi);
        unsigned long long *p_ind = ind.memptr();
        const dtype *pi = yi->memptr();

        dtype *p_grid_srt = new dtype[ny];
        if (!sorted)
            for (auto i = 0ULL; i < ny; ++i)
                p_grid_srt[i] = pi[p_ind[i]];
        const dtype *p_grid = sorted ? pi : p_grid_srt;

        dtype *p_diff = new dtype[ny];
        *p_diff = one;
        for (auto i = 1ULL; i < ny; ++i)
            p_diff[i] = one / (p_grid[i] - p_grid[i - 1]);

        const dtype *po = yo->memptr();
        for (auto i = 0ULL; i < my; ++i)
        {
            unsigned long long ip = 0ULL, in = 0ULL; // Indices for reading the input data
            dtype val = po[i], wp = one, wn = zero;  // Relative weights for interpolation
            while (ip < ny && p_grid[ip] <= val)
                ip++;
            if (ip == ny)
                in = --ip;
            else if (ip != 0)
            {
                in = ip--;
                wp = (p_grid[in] - val) * p_diff[in];
                wp = wp > one ? one : wp;
                wp = wp < zero ? zero : wp;
                wn = one - wp;
            }
            i_yp_vec[i] = p_ind[ip], i_yn_vec[i] = p_ind[in], yp_vec[i] = wp, yn_vec[i] = wn;
        }
        delete[] p_diff;
        delete[] p_grid_srt;
        ind.reset();
    }

    // Interpolate the input data and write to output memory
    const dtype *p_input = input->memptr();
    for (int ie = 0; ie < int(ne); ie++)
    {
        dtype *p_output = output->slice_memptr(ie);
        unsigned long long offset = ie * nxy;
        for (auto ix = 0ULL; ix < mx; ++ix)
        {
            for (auto iy = 0ULL; iy < my; ++iy)
            {
                unsigned long long iA = offset + i_xp_vec[ix] * ny + i_yp_vec[iy];
                unsigned long long iB = offset + i_xn_vec[ix] * ny + i_yp_vec[iy];
                unsigned long long iC = offset + i_xp_vec[ix] * ny + i_yn_vec[iy];
                unsigned long long iD = offset + i_xn_vec[ix] * ny + i_yn_vec[iy];

                dtype wA = xp_vec[ix] * yp_vec[iy];
                dtype wB = xn_vec[ix] * yp_vec[iy];
                dtype wC = xp_vec[ix] * yn_vec[iy];
                dtype wD = xn_vec[ix] * yn_vec[iy];

                *p_output++ = wA * p_input[iA] + wB * p_input[iB] + wC * p_input[iC] + wD * p_input[iD];
            }
        }
    }

    delete[] i_xp_vec;
    delete[] i_xn_vec;
    delete[] i_yp_vec;
    delete[] i_yn_vec;
    delete[] xp_vec;
    delete[] xn_vec;
    delete[] yp_vec;
    delete[] yn_vec;

    return "";
}
template std::string quadriga_lib::interp(const arma::Cube<float> *input, const arma::Col<float> *xi, const arma::Col<float> *yi,
                                          const arma::Col<float> *xo, const arma::Col<float> *yo, arma::Cube<float> *output);
template std::string quadriga_lib::interp(const arma::Cube<double> *input, const arma::Col<double> *xi, const arma::Col<double> *yi,
                                          const arma::Col<double> *xo, const arma::Col<double> *yo, arma::Cube<double> *output);

// 1D linear interpolation
template <typename dtype>
std::string quadriga_lib::interp(const arma::Mat<dtype> *input, const arma::Col<dtype> *xi,
                                 const arma::Col<dtype> *xo, arma::Mat<dtype> *output)
{
    const unsigned long long nx = input->n_rows, ne = input->n_cols;
    const unsigned long long mx = xo->n_elem;

    if (input->n_elem == 0 || xi->n_elem != nx)
        return "Data dimensions must match the given number of sample points.";

    if (mx == 0)
        return "Data dimensions must match the given number of sample points.";

    if (output->n_rows != mx || output->n_cols != ne)
        output->set_size(mx, ne);

    // Reinterpret input matrices as cubes
    const arma::Cube<dtype> input_cube = arma::Cube<dtype>(const_cast<dtype *>(input->memptr()), 1, nx, ne, false, true);
    arma::Cube<dtype> output_cube = arma::Cube<dtype>(output->memptr(), 1, mx, ne, false, true);
    arma::Col<dtype> y(1);

    // Call 2D linear interpolation function
    std::string error_message = quadriga_lib::interp(&input_cube, xi, &y, xo, &y, &output_cube);
    return error_message;
}

template std::string quadriga_lib::interp(const arma::Mat<float> *input, const arma::Col<float> *xi, const arma::Col<float> *xo, arma::Mat<float> *output);
template std::string quadriga_lib::interp(const arma::Mat<double> *input, const arma::Col<double> *xi, const arma::Col<double> *xo, arma::Mat<double> *output);

// Read Wavefront .obj file
template <typename dtype>
unsigned quadriga_lib::obj_file_read(std::string fn, arma::Mat<dtype> *mesh, arma::Mat<dtype> *mtl_prop, arma::Mat<dtype> *vert_list,
                                     arma::Mat<unsigned> *face_ind, arma::Col<unsigned> *obj_ind, arma::Col<unsigned> *mtl_ind)
{
    // Open file for reading
    std::ifstream fileR = std::ifstream(fn, std::ios::in);
    if (!fileR.is_open())
        throw std::invalid_argument("Error opening file.");

    // Obtain the number of faces and vertices from the file
    unsigned n_vert = 0, n_faces = 0;
    std::string line;
    while (std::getline(fileR, line))
        if (line.length() > 2 && line.at(0) == 118 && line.at(1) == 32) // Line starts with "v "
            ++n_vert;
        else if (line.length() > 2 && line.at(0) == 102) // Line starts with "f "
            ++n_faces;

    // Stop here if no other outputs are needed
    if (n_vert == 0 || n_faces == 0)
    {
        fileR.close();
        return 0;
    }

    if (mesh == nullptr && mtl_prop == nullptr && vert_list == nullptr && face_ind == nullptr && obj_ind == nullptr && mtl_ind == nullptr)
    {
        fileR.close();
        return n_faces;
    }

    // Define a struct to store the material properties
    struct MaterialProp
    {
        std::string name;  // Material name
        double a, b, c, d; // Electromagnetic properties
        double att;        // Additional fixed  attenuation in dB
        unsigned index;    // Material index
    };

    // Add default material data, See: Rec. ITU-R P.2040-1, Table 3
    std::vector<MaterialProp> mtl_lib;
    mtl_lib.push_back({"Concrete", 5.31, 0.0, 0.0326, 0.8095, 0.0, 0});
    mtl_lib.push_back({"Brick", 3.75, 0.0, 0.038, 0.0, 0.0, 0});
    mtl_lib.push_back({"Plasterboard", 2.94, 0.0, 0.0116, 0.7076, 0.0, 0});
    mtl_lib.push_back({"Wood", 1.99, 0.0, 0.0047, 1.0718, 0.0, 0});
    mtl_lib.push_back({"Glass", 6.27, 0.0, 0.0043, 1.1925, 0.0, 0});
    mtl_lib.push_back({"Chipboard", 2.58, 0.0, 0.0217, 0.78, 0.0, 0});
    mtl_lib.push_back({"Metal", 1.0, 0.0, 1.0e7, 0.0, 0.0, 0});
    mtl_lib.push_back({"Ground_dry", 3.0, 0.0, 0.00015, 2.52, 0.0, 0});
    mtl_lib.push_back({"Ground_medium", 15.0, -0.1, 0.035, 1.63, 0.0, 0});
    mtl_lib.push_back({"Ground_wet", 30.0, -0.4, 0.15, 1.3, 0.0, 0});
    mtl_lib.push_back({"Vegetation", 1.0, 0.0, 1.0e-4, 1.1, 0.0, 0});     // Rec. ITU-R P.833-9, Figure 2
    mtl_lib.push_back({"Water", 80.0, 0.0, 0.2, 2.0, 0.0, 0});            // Rec. ITU-R P.527-3, Figure 1
    mtl_lib.push_back({"Ice", 3.0, 0.0, 3.0e-4, 0.7, 0.0, 0});            // Rec. ITU-R P.527-3, Figure 1
    mtl_lib.push_back({"IRR_glass", 6.27, 0.0, 0.0043, 1.1925, 23.0, 0}); // 3GPP TR 38.901 V17.0.0, Table 7.4.3-1: Material penetration losses

    // Reset the file pointer to the beginning of the file
    fileR.clear(); // Clear any flags
    fileR.seekg(0, std::ios::beg);

    // Local data
    unsigned i_vert = 0, i_face = 0, j_face = 0, i_object = 0, i_mtl = 0; // Counters for vertices, faces, objects, materials
    double aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0;            // Default material properties
    unsigned iM = 0;                                                      // Material index
    bool simple_face_format = true;                                       // Selector for face format

    // Obtain memory for the vertex list
    dtype *p_vert;
    if (vert_list == nullptr)
        p_vert = new dtype[n_vert * 3];
    else if (vert_list->n_rows != (unsigned long long)n_vert || vert_list->n_cols != 3ULL)
    {
        vert_list->set_size((unsigned long long)n_vert, 3ULL);
        p_vert = vert_list->memptr();
    }
    else
        p_vert = vert_list->memptr();

    // Obtain memory for face indices
    unsigned *p_face_ind;
    if (face_ind == nullptr)
        p_face_ind = new unsigned[n_faces * 3];
    else if (face_ind->n_rows != (unsigned long long)n_faces || face_ind->n_cols != 3ULL)
    {
        face_ind->set_size((unsigned long long)n_faces, 3ULL);
        p_face_ind = face_ind->memptr();
    }
    else
        p_face_ind = face_ind->memptr();

    // Set size of "mtl_prop"
    if (mtl_prop != nullptr && (mtl_prop->n_rows != (unsigned long long)n_faces || mtl_prop->n_cols != 5ULL))
        mtl_prop->set_size((unsigned long long)n_faces, 5ULL);
    dtype *p_mtl_prop = mtl_prop == nullptr ? nullptr : mtl_prop->memptr();

    // Set size of "mtl_ind"
    if (mtl_ind != nullptr && mtl_ind->n_elem != (unsigned long long)n_faces)
        mtl_ind->set_size((unsigned long long)n_faces);
    unsigned *p_mtl_ind = mtl_ind == nullptr ? nullptr : mtl_ind->memptr();

    // Set size of "obj_ind"
    if (obj_ind != nullptr && obj_ind->n_elem != (unsigned long long)n_faces)
        obj_ind->set_size((unsigned long long)n_faces);
    unsigned *p_obj_ind = obj_ind == nullptr ? nullptr : obj_ind->memptr();

    // Process file
    while (std::getline(fileR, line))
    {
        // Read vertex
        if (line.length() > 2 && line.at(0) == 118 && line.at(1) == 32) // Line starts with "v "
        {
            if (i_vert >= n_vert)
                throw std::invalid_argument("Error reading vertex data.");

            double x, y, z;
            std::sscanf(line.c_str(), "v %lf %lf %lf", &x, &y, &z);
            p_vert[i_vert] = (dtype)x;
            p_vert[i_vert + n_vert] = (dtype)y;
            p_vert[i_vert++ + 2ULL * n_vert] = (dtype)z;
        }

        // Read face
        else if (line.length() > 2 && line.at(0) == 102) // Line starts with "f "
        {
            if (i_face >= n_faces)
                throw std::invalid_argument("Error reading face data.");

            // Read face indices from file (1-based)
            int a = 0, b = 0, c = 0, d = 0;
            if (simple_face_format)
            {
                sscanf(line.c_str(), "f %d %d %d %d", &a, &b, &c, &d);
                simple_face_format = b != 0;
            }
            if (!simple_face_format)
                sscanf(line.c_str(), "f %d%*[/0-9] %d%*[/0-9] %d%*[/0-9] %d", &a, &b, &c, &d);

            if (a == 0 || b == 0 || c == 0)
                throw std::invalid_argument("Error reading face data.");

            if (d != 0)
                throw std::invalid_argument("Mesh is not in triangularized form.");

            // Store current material properties
            if (p_mtl_prop != nullptr)
                p_mtl_prop[i_face] = (dtype)aM,
                p_mtl_prop[i_face + n_faces] = (dtype)bM,
                p_mtl_prop[i_face + 2 * n_faces] = (dtype)cM,
                p_mtl_prop[i_face + 3 * n_faces] = (dtype)dM,
                p_mtl_prop[i_face + 4 * n_faces] = (dtype)attM;

            if (p_mtl_ind != nullptr)
                p_mtl_ind[i_face] = iM;

            // Store face indices (0-based)
            p_face_ind[i_face] = (unsigned)a - 1;
            p_face_ind[i_face + n_faces] = (unsigned)b - 1;
            p_face_ind[i_face++ + 2 * n_faces] = (unsigned)c - 1;
        }

        // Read objects ids (= connected faces)
        // - Object name is written to the OBJ file before vertices, materials and faces
        else if (line.length() > 2 && line.at(0) == 111) // Line starts with "o "
        {
            if (p_obj_ind != nullptr)
                for (unsigned i = j_face; i < i_face; ++i)
                    p_obj_ind[i] = i_object;

            // Reset current material
            aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0, iM = 0;
            j_face = i_face;
            ++i_object;
        }

        // Read and set material properties
        // - Material names are written before face indices
        else if (line.length() > 7 && line.substr(0, 6).compare("usemtl") == 0) // Line contains material definition
        {
            std::string mtl_name = line.substr(7, 255);                 // Name in OBJ File
            aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0, iM = 0; // Reset current material
            int found = -1;

            // Try to find the material name in the material library
            for (int n = 0; n < (int)mtl_lib.size(); ++n)
                if (mtl_lib[n].name.compare(mtl_name) == 0)
                    aM = mtl_lib[n].a, bM = mtl_lib[n].b, cM = mtl_lib[n].c, dM = mtl_lib[n].d, attM = mtl_lib[n].att, iM = mtl_lib[n].index, found = n;

            if (found == -1) // Add new material
            {
                sscanf(mtl_name.c_str(), "%*[a-zA-Z0-9. ]::%lf:%lf:%lf:%lf:%lf", &aM, &bM, &cM, &dM, &attM);
                if (aM == 0.0)
                    mtl_lib.push_back({mtl_name, 1.0, 0.0, 0.0, 0.0, 0.0, 0}); // Air
                else
                    mtl_lib.push_back({mtl_name, aM, bM, cM, dM, attM, 0});
                found = (int)mtl_lib.size() - 1;
            }

            if (iM == 0) // Increase material counter
                iM = ++i_mtl, mtl_lib[found].index = i_mtl;
        }
    }

    // Set the object ID of the last object
    i_object = i_object == 0 ? 1 : i_object; // Single unnamed object
    if (p_obj_ind != nullptr)
        for (unsigned i = j_face; i < i_face; ++i)
            p_obj_ind[i] = i_object;

    // Calculate the triangle mesh from vertices and faces
    if (mesh != nullptr)
    {
        if (mesh->n_rows != (unsigned long long)n_faces || mesh->n_cols != 9ULL)
            mesh->set_size((unsigned long long)n_faces, 9ULL);
        dtype *p_mesh = mesh->memptr();

        for (unsigned n = 0; n < n_faces; ++n)
        {
            unsigned a = p_face_ind[n],
                     b = p_face_ind[n + n_faces],
                     c = p_face_ind[n + 2 * n_faces];

            if (a > n_vert || b > n_vert || c > n_vert)
                throw std::invalid_argument("Error assembling triangle mesh.");

            p_mesh[n] = p_vert[a];
            p_mesh[n + n_faces] = p_vert[a + n_vert];
            p_mesh[n + 2 * n_faces] = p_vert[a + 2 * n_vert];
            p_mesh[n + 3 * n_faces] = p_vert[b];
            p_mesh[n + 4 * n_faces] = p_vert[b + n_vert];
            p_mesh[n + 5 * n_faces] = p_vert[b + 2 * n_vert];
            p_mesh[n + 6 * n_faces] = p_vert[c];
            p_mesh[n + 7 * n_faces] = p_vert[c + n_vert];
            p_mesh[n + 8 * n_faces] = p_vert[c + 2 * n_vert];
        }
    }

    // Clean up and return
    mtl_lib.clear();

    if (vert_list == nullptr)
        delete[] p_vert;

    if (face_ind == nullptr)
        delete[] p_face_ind;

    fileR.close();

    return n_faces;
}

template unsigned quadriga_lib::obj_file_read(std::string fn, arma::Mat<float> *mesh, arma::Mat<float> *mtl_prop, arma::Mat<float> *vert_list,
                                              arma::Mat<unsigned> *face_ind, arma::Col<unsigned> *obj_ind, arma::Col<unsigned> *mtl_ind);
template unsigned quadriga_lib::obj_file_read(std::string fn, arma::Mat<double> *mesh, arma::Mat<double> *mtl_prop, arma::Mat<double> *vert_list,
                                              arma::Mat<unsigned> *face_ind, arma::Col<unsigned> *obj_ind, arma::Col<unsigned> *mtl_ind);

// Subdivide triangles into smaller triangles
template <typename dtype>
unsigned long long quadriga_lib::subdivide_triangles(unsigned long long n_div, const arma::Mat<dtype> *triangles_in, arma::Mat<dtype> *triangles_out)
{
    if (n_div == 0ULL)
        throw std::invalid_argument("Input 'n_div' cannot be 0.");

    if (triangles_in == nullptr || triangles_in->n_elem == 0ULL)
        throw std::invalid_argument("Input 'triangles_in' cannot be NULL.");

    if (triangles_in->n_cols != 9ULL)
        throw std::invalid_argument("Input 'triangles_in' must have 9 columns.");

    if (triangles_out == nullptr)
        throw std::invalid_argument("Output 'triangles_out' cannot be NULL.");

    unsigned long long n_triangles_in = triangles_in->n_rows;
    unsigned long long n_triangles_out = n_triangles_in * n_div * n_div;

    if (triangles_out->n_cols != 9ULL || triangles_out->n_rows != n_triangles_out)
        triangles_out->set_size(n_triangles_out, 9ULL);

    // Process each triangle
    unsigned long long cnt = 0ULL;                        // Counter
    dtype stp = dtype(1.0) / dtype(n_div);                // Step size
    const dtype *p_triangles_in = triangles_in->memptr(); // Pointer to input memory
    dtype *p_triangles_out = triangles_out->memptr();     // Pointer to output memory

    for (auto n = 0ULL; n < n_triangles_in; ++n)
    {
        // Read current triangle vertices
        dtype v1x = p_triangles_in[n];
        dtype v1y = p_triangles_in[n + n_triangles_in];
        dtype v1z = p_triangles_in[n + 2ULL * n_triangles_in];
        dtype e12x = p_triangles_in[n + 3ULL * n_triangles_in] - v1x;
        dtype e12y = p_triangles_in[n + 4ULL * n_triangles_in] - v1y;
        dtype e12z = p_triangles_in[n + 5ULL * n_triangles_in] - v1z;
        dtype e13x = p_triangles_in[n + 6ULL * n_triangles_in] - v1x;
        dtype e13y = p_triangles_in[n + 7ULL * n_triangles_in] - v1y;
        dtype e13z = p_triangles_in[n + 8ULL * n_triangles_in] - v1z;

        for (auto u = 0ULL; u < n_div; ++u)
        {
            dtype ful = dtype(u) * stp;
            dtype fuu = ful + stp;

            for (auto v = 0ULL; v < n_div - u; ++v)
            {
                dtype fvl = dtype(v) * stp;
                dtype fvu = fvl + stp;

                // Lower triangle first vertex
                p_triangles_out[cnt] = v1x + fvl * e12x + ful * e13x;                          // w1x
                p_triangles_out[cnt + n_triangles_out] = v1y + fvl * e12y + ful * e13y;        // w1y
                p_triangles_out[cnt + 2ULL * n_triangles_out] = v1z + fvl * e12z + ful * e13z; // w1z

                // Lower triangle second vertex
                p_triangles_out[cnt + 3ULL * n_triangles_out] = v1x + fvu * e12x + ful * e13x; // w2x
                p_triangles_out[cnt + 4ULL * n_triangles_out] = v1y + fvu * e12y + ful * e13y; // w2y
                p_triangles_out[cnt + 5ULL * n_triangles_out] = v1z + fvu * e12z + ful * e13z; // w2z

                // Lower triangle third vertex
                p_triangles_out[cnt + 6ULL * n_triangles_out] = v1x + fvl * e12x + fuu * e13x; // w2x
                p_triangles_out[cnt + 7ULL * n_triangles_out] = v1y + fvl * e12y + fuu * e13y; // w2y
                p_triangles_out[cnt + 8ULL * n_triangles_out] = v1z + fvl * e12z + fuu * e13z; // w2z
                cnt++;

                if (v < n_div - u - 1ULL)
                {
                    // Upper triangle first vertex
                    p_triangles_out[cnt] = v1x + fvl * e12x + fuu * e13x;                          // w1x
                    p_triangles_out[cnt + n_triangles_out] = v1y + fvl * e12y + fuu * e13y;        // w1y
                    p_triangles_out[cnt + 2ULL * n_triangles_out] = v1z + fvl * e12z + fuu * e13z; // w1z

                    // Upper triangle second vertex
                    p_triangles_out[cnt + 6ULL * n_triangles_out] = v1x + fvu * e12x + fuu * e13x; // w2x
                    p_triangles_out[cnt + 7ULL * n_triangles_out] = v1y + fvu * e12y + fuu * e13y; // w2y
                    p_triangles_out[cnt + 8ULL * n_triangles_out] = v1z + fvu * e12z + fuu * e13z; // w2z

                    // Upper triangle third vertex
                    p_triangles_out[cnt + 3ULL * n_triangles_out] = v1x + fvu * e12x + ful * e13x; // w2x
                    p_triangles_out[cnt + 4ULL * n_triangles_out] = v1y + fvu * e12y + ful * e13y; // w2y
                    p_triangles_out[cnt + 5ULL * n_triangles_out] = v1z + fvu * e12z + ful * e13z; // w2z
                    ++cnt;
                }
            }
        }
    }
    return n_triangles_out;
}

template unsigned long long quadriga_lib::subdivide_triangles(unsigned long long n_div, const arma::Mat<float> *triangles_in, arma::Mat<float> *triangles_out);
template unsigned long long quadriga_lib::subdivide_triangles(unsigned long long n_div, const arma::Mat<double> *triangles_in, arma::Mat<double> *triangles_out);
