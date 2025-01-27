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
#include <sstream>
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

// Cross product
template <typename dtype>
static inline void crossp(dtype vx, dtype vy, dtype vz,    // Vector V
                          dtype kx, dtype ky, dtype kz,    // Vector K
                          dtype *rx, dtype *ry, dtype *rz, // Result V x K
                          bool normalize_output = false)
{
    *rx = vy * kz - vz * ky; // x-component
    *ry = vz * kx - vx * kz; // y-component
    *rz = vx * ky - vy * kx; // z-component

    if (normalize_output)
    {
        dtype scl = (dtype)1.0 / std::sqrt(*rx * *rx + *ry * *ry + *rz * *rz);
        *rx *= scl, *ry *= scl, *rz *= scl;
    }
}

// Dot product
template <typename dtype>
static inline dtype dotp(dtype vx, dtype vy, dtype vz, // Vector V
                         dtype kx, dtype ky, dtype kz, // Vector K
                         bool normalize = false)       // Option to normalize
{
    dtype dot = vx * kx + vy * ky + vz * kz; // Dot product calculation

    if (normalize)
    {
        dtype v_mag = std::sqrt(vx * vx + vy * vy + vz * vz); // Magnitude of V
        dtype k_mag = std::sqrt(kx * kx + ky * ky + kz * kz); // Magnitude of K
        if (v_mag > 0 && k_mag > 0)                           // Normalize the dot product
            dot /= (v_mag * k_mag);
    }

    return dot;
}

// Line-Triangle intersect in 2D coordinates
// - Triangle is defined by vertices P0(0,0), P1(x1,0) and P2(x2,y2)
// - The line segment is given by O(ox,oy) -> D(dx,dy)
// - We consider special rules for “on-edge” tolerance, co-linearity, etc. as follows:
//   1. If either point O or D lies within the triangle and the other lies outside or at the boundary
//      (i.e. the line crosses one edge/vertex or lies on that edge/vertex), it counts as a hit
//   2. If the entire line segment lies within the triangle (i.e. both points are inside), it counts as a hit
//   3. If one point O or D lies "on the edge", i.e. is within a tolerance distance (usually 0.5 mm)
//      from any edge and the other point is outside the triangle, it counts as a miss
//   4. If both points lie on strictly different edges (within the tolerance), excluding vertices,
//      the segment passes through the triangle and it counts as a hit
//   5. If one point O/D lies on a vertex P0/P1/P2 and the other lies on the opposite edge => hit
//   6. If both points lie outside the triangle, but the line segment passes through the triangle
//      (i.e. the line crosses two edges), it counts as a hit. Touching corners count as miss.
//   7. If the line is co-linear with an edge, i.e. at least one point is on the edge within the
//       tolerance and the line is parallel to the edge, there are 3 options:
//          7a. If the edges only "touch" at the end points (i.e. do not overlap by more than the tolerance), it counts as a miss
//          7b. If the edge and the line point in the same direction, it counts as a hit
//          7c. If the edge and the line point in opposite directions, it counts as a miss
// - Returns 0 for a miss or the rule number that caused the hit otherwise
// - Returns -1 if the triangle is degenerate, i.e. has a zero-sized area
template <typename dtype>
static inline int line_triangle_intersect_2D(dtype x1, dtype x2, dtype y2,    // Triangle
                                             dtype ox, dtype oy,              // Line origin point O
                                             dtype dx, dtype dy,              // Line destination point D
                                             dtype tolerance = (dtype)0.0005) // Intersect tolerance in m
{
    const dtype eps = (dtype)1.0e-6; // Numeric tolerance
    const dtype z0 = (dtype)0.0;     // Zero

    // Returns cross product of vectors A(ax,ay) x B(bx,by).
    auto crossp2D = [&](dtype ax, dtype ay, dtype bx, dtype by) -> dtype
    {
        return ax * by - ay * bx;
    };

    // Compute twice the area of the triangle with edges A(ax,ay) -> B(bx,by) -> C(cx,cy).
    auto area2D = [&](dtype ax, dtype ay, dtype bx, dtype by, dtype cx, dtype cy) -> dtype
    {
        return std::fabs(crossp2D((bx - ax), (by - ay), (cx - ax), (cy - ay)));
    };

    // Check if the triangle is degenerate
    const dtype A = area2D(z0, z0, x1, z0, x2, y2);
    if (A < eps)
        return -1;

    // Distance from point P(px,py) to the point A(ax,ay)
    auto PointToPointDistance = [&](dtype px, dtype py, dtype ax, dtype ay) -> dtype
    {
        const dtype APx = px - ax, APy = py - ay; // Vector A->P: (px-ax, py-ay)
        return std::sqrt(APx * APx + APy * APy);  // Length
    };

    // Distance from point P(px,py) to the line segment A(ax,ay) -> B(bx,by)
    auto PointToSegmentDistance = [&](dtype px, dtype py, dtype ax, dtype ay, dtype bx, dtype by) -> dtype
    {
        const dtype APx = px - ax, APy = py - ay;    // Vector A->P: (px-ax, py-ay),
        const dtype ABx = bx - ax, ABy = by - ay;    // Vector A->B: (bx-ax, by-ay)
        const dtype AB_len2 = ABx * ABx + ABy * ABy; // length^2 of segment
        dtype x = z0, y = z0;

        if (AB_len2 < eps) // A and B are the same point. Return distance from P to A
        {
            x = px - ax, y = py - ay;
            return std::sqrt(x * x + y * y);
        }

        // Project AP onto AB to find parameter t along AB
        const dtype t = (APx * ABx + APy * ABy) / AB_len2;

        if (t < z0) // If t < 0 => nearest to A
            x = px - ax, y = py - ay;
        else if (t > (dtype)1.0) // If t > 1 => nearest to B.
            x = px - bx, y = py - by;
        else // Nearest is the projection onto the segment
            x = px - (ax + t * ABx), y = py - (ay + t * ABy);

        return std::sqrt(x * x + y * y);
    };

    // Checks if the line O(ox,oy) -> D(dx,dy) is co-linear with edge A(ax,ay) -> B(bx,by) as per rule #7:
    //   7a. If the edges only "touch" at the end points (i.e. do not overlap by more than the tolerance), it counts as a miss
    //   7b. If the edge and the line point in the same direction, it counts as a hit
    //   7c. If the edge and the line point in opposite directions, it counts as a miss
    // Returns: 0 = not co-linear (e.g. points not near-edge, line and edge not parallel)
    //          1 = co-linear (case 7b, overlapping and pointing in same direction)
    //         -1 = MISS according to case 7a (end-points touching, but no overlap)
    //         -2 = MISS according to case 7c (opposite directions)
    auto line_colinear_with = [&](dtype ax, dtype ay, dtype bx, dtype by) -> int
    {
        // 1) Are O or D on this edge within tolerance?
        dtype distO = PointToSegmentDistance(ox, oy, ax, ay, bx, by);
        dtype distD = PointToSegmentDistance(dx, dy, ax, ay, bx, by);
        if (distO >= tolerance && distD >= tolerance)
            return 0; // not co-linear if neither endpoint is near the edge

        // 2) Check if the vectors are parallel:
        dtype Lx = (dx - ox), Ly = (dy - oy); // Line vector
        dtype Ex = (bx - ax), Ey = (by - ay); // Edge vector
        if (std::fabs(crossp2D(Ex, Ey, Lx, Ly)) > eps)
            return 0; // Not parallel => not co-linear

        // 3) Check direction via dot product
        if ((Ex * Lx + Ey * Ly) < z0)
            return -2; // opposite direction => not co-linear

        // 4) Check for touching end points
        // Here we know already that the lines are parallel (2), point in the same direction (3) and O or D are on-edge (1)
        if (PointToPointDistance(dx, dy, ax, ay) < tolerance)
            return -1; // Line destination D is co-located with edge start A => no overlap possible
        if (PointToPointDistance(ox, oy, bx, by) < tolerance)
            return -1; // Line origin O is co-located with edge end point B => no overlap possible

        // If we reached this point, all previous checks passed, hence only case 7b is left
        return 1;
    };

    // Segment intersection test according to rule 6
    // Returns true if line O(ox,oy) -> D(dx,dy) strictly intersects edge A(ax,ay) -> B(bx,by)
    auto segmentsIntersect = [&](dtype ax, dtype ay, dtype bx, dtype by) -> bool
    {
        // 1) Check if the line (O->D) crosses through a vertex. This case is handled by rules (1) or (5).
        //    Vertex crossings are ambiguous since they can produce 0/1/2 hits. We remove this option here.
        dtype distA = PointToSegmentDistance(ax, ay, ox, oy, dx, dy);
        dtype distB = PointToSegmentDistance(bx, by, ox, oy, dx, dy);
        if (distA < tolerance || distB < tolerance)
            return false; // end points A or B are on the line (O->D) => no intersection (handled by rules 1 or 5)

        // 2) Using orientation or cross-product approach to find intersections
        auto o1 = crossp2D((dx - ox), (dy - oy), (ax - ox), (ay - oy));
        auto o2 = crossp2D((dx - ox), (dy - oy), (bx - ox), (by - oy));
        auto o3 = crossp2D((bx - ax), (by - ay), (ox - ax), (oy - ay));
        auto o4 = crossp2D((bx - ax), (by - ay), (dx - ax), (dy - ay));

        auto sign_of = [&](dtype v) -> int
        {
            return (v > 0) ? 1 : ((v < 0) ? -1 : 0);
        };

        int s1 = sign_of(o1), s2 = sign_of(o2);
        int s3 = sign_of(o3), s4 = sign_of(o4);

        // Intersection condition:
        if (s1 != s2 && s3 != s4)
            return true;

        // No crossing
        return false;
    };

    // Classify a point as either:
    // -1 = on vertex P0 = (0,0)
    // -2 = on vertex P1 = (x1,0)
    // -3 = on vertex P2 = (x2,y2)
    //  0 = outside
    //  1 = inside
    //  2 = on edge 0 (but not on P0 or P1)
    //  3 = on edge 1 (but not on P1 or P2)
    //  4 = on edge 2 (but not on P2 or P0)
    auto classifyPoint = [&](dtype px, dtype py) -> int
    {
        // 1) Check if on vertex:
        dtype dist = PointToPointDistance(px, py, z0, z0); // P0 = (0,0)
        if (dist < tolerance)
            return -1;
        dist = PointToPointDistance(px, py, x1, z0); // P1 = (x1,0)
        if (dist < tolerance)
            return -2;
        dist = PointToPointDistance(px, py, x2, y2); // P2 = (x2,y2)
        if (dist < tolerance)
            return -3;

        // 2) Check if on-edge:
        dist = PointToSegmentDistance(px, py, z0, z0, x1, z0); // Edge 0: P0->P1
        if (dist < tolerance)
            return 2;
        dist = PointToSegmentDistance(px, py, x1, z0, x2, y2); // Edge 1: P1->P2
        if (dist < tolerance)
            return 3;
        dist = PointToSegmentDistance(px, py, x2, y2, z0, z0); // Edge 2: P2->P0
        if (dist < tolerance)
            return 4;

        // 3) Check if inside (strictly ignoring edges):
        //    We'll do area-sum approach. We'll say "inside" if areaP0P1P2 == area(P0,P1,P) + area(P1,P2,P) + area(P2,P0,P)
        //    with a small epsilon. We'll treat zero on an edge as "outside" to be consistent with the prior "on-edge" check.
        const dtype A1 = area2D(px, py, x1, z0, x2, y2);
        const dtype A2 = area2D(z0, z0, px, py, x2, y2);
        const dtype A3 = area2D(z0, z0, x1, z0, px, py);
        const dtype sumA = A1 + A2 + A3;

        if (std::fabs(sumA - A) < eps && A1 > eps && A2 > eps && A3 > eps)
            return 1; // inside

        // 4) Otherwise => outside
        return 0;
    };
    int cO = classifyPoint(ox, oy);
    int cD = classifyPoint(dx, dy);

    // Rule (7): Check if the line O(ox,oy) -> D(dx,dy) is co-linear with any edge of the triangle:
    int cP0 = line_colinear_with(z0, z0, x1, z0); // Edge 0: P0->P1
    int cP1 = line_colinear_with(x1, z0, x2, y2); // Edge 1: P1->P2
    int cP2 = line_colinear_with(x2, y2, z0, z0); // Edge 2: P2->P0

    // The line can only be co-linear with one of the 3 edges, hence only one vale can be != 0
    if (cP0 == 1 || cP1 == 1 || cP2 == 1) // Case 7b triggered for one of the edges
        return 7;                         // Return HIT for rule 7
    if (cP0 < 0 || cP1 < 0 || cP2 < 0)    // Case 7a or 7c triggered for one of the edges
        return 0;                         // Return MISS, skip all further checks

    // Rule (2): Both points inside
    if (cO == 1 && cD == 1)
        return 2;

    // Rule (1): One point inside, the other outside or on edge/vertex
    if (cO == 1 || cD == 1)
        return 1;

    // Rule (3): "If one point is on the edge/vertex and the other is outside => MISS"
    if ((cO >= 2 && cD == 0) || (cO == 0 && cD >= 2))
        return 0;
    if ((cO < 0 && cD == 0) || (cO == 0 && cD < 0))
        return 0;

    // Rule (4): Both points on different edges
    if (cO >= 2 && cD >= 2 && cO != cD)
        return 4;

    // Rule (5): One point on-vertex, the other on the opposite edge
    if ((cO == -1 && cD == 3) || (cO == -2 && cD == 4) || (cO == -3 && cD == 2))
        return 5;
    if ((cD == -1 && cO == 3) || (cD == -2 && cO == 4) || (cD == -3 && cO == 2))
        return 5;

    // Rule (6): Both points lie outside, but line segment passes through the triangle
    if (cO == 0 && cD == 0)
    {
        int countIntersections = 0;
        if (segmentsIntersect(z0, z0, x1, z0)) // Edge 0: P0->P1
            ++countIntersections;
        if (segmentsIntersect(x1, z0, x2, y2)) // Edge 1: P1->P2
            ++countIntersections;
        if (segmentsIntersect(x2, y2, z0, z0)) // Edge 2: P2->P0
            ++countIntersections;

        // If the line truly passes through the interior, we typically expect 2 intersections.
        // (Tangential or 1 intersection => no pass.)
        if (countIntersections >= 2)
            return 6;
    }

    // If we somehow get here (e.g. missed some edge case?), default to no intersection
    return 0;
}

// Rotate a vector around an arbitrary axis
static inline void rotate_vector_around_axis(double vx, double vy, double vz,    // The vector to rotate
                                             double kx, double ky, double kz,    // The axis of rotation
                                             double theta,                       // Rotation angle in radians
                                             double *rx, double *ry, double *rz, // Result
                                             bool k_is_normalized = false)
{
    // Step 1: Precompute values
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    if (!k_is_normalized)
    {
        double scl = kx * kx + ky * ky + kz * kz;
        scl = 1.0 / std::sqrt(scl);
        kx *= scl, ky *= scl, kz *= scl;
    }

    double dot = vx * kx + vy * ky + vz * kz; // Dot product of V and K

    // Step 2: Rodrigues' rotation formula components
    *rx = vx * cos_theta + (ky * vz - kz * vy) * sin_theta + kx * dot * (1.0 - cos_theta);
    *ry = vy * cos_theta + (kz * vx - kx * vz) * sin_theta + ky * dot * (1.0 - cos_theta);
    *rz = vz * cos_theta + (kx * vy - ky * vx) * sin_theta + kz * dot * (1.0 - cos_theta);
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
                              arma::Mat<dtype> *path_angles, std::vector<arma::Mat<dtype>> *path_coord)
{
    if (no_interact == nullptr)
        throw std::invalid_argument("Input 'no_interact' cannot be NULL.");

    size_t n_path = (size_t)no_interact->n_elem;

    // Calculate the total number of interactions
    unsigned interact_cnt = 0;
    const unsigned *p_interact = no_interact->memptr();
    for (size_t i = 0; i < n_path; ++i)
        interact_cnt += p_interact[i];

    size_t n_interact = (size_t)interact_cnt;

    if (interact_coord == nullptr || interact_coord->n_rows != 3)
        throw std::invalid_argument("Input 'interact_coord' must have 3 rows.");

    if (interact_coord->n_cols != n_interact)
        throw std::invalid_argument("Number of columns of 'interact_coord' must match the sum of 'no_interact'.");

    constexpr dtype zero = dtype(0.0);
    constexpr dtype half = dtype(0.5);
    constexpr dtype los_limit = dtype(1.0e-4);

    // Set the output size
    if (path_length != nullptr && path_length->n_elem != n_path)
        path_length->set_size(n_path);
    if (fbs_pos != nullptr && (fbs_pos->n_rows != 3 || fbs_pos->n_cols != n_path))
        fbs_pos->set_size(3, n_path);
    if (lbs_pos != nullptr && (lbs_pos->n_rows != 3 || lbs_pos->n_cols != n_path))
        lbs_pos->set_size(3, n_path);
    if (path_angles != nullptr && (path_angles->n_rows != n_path || path_angles->n_cols != 4))
        path_angles->set_size(n_path, 4);
    if (path_coord != nullptr && path_coord->size() != n_path)
        path_coord->resize(n_path);

    // Get pointers
    const dtype *p_coord = interact_coord->memptr();
    dtype *p_length = path_length == nullptr ? nullptr : path_length->memptr();
    dtype *p_fbs = fbs_pos == nullptr ? nullptr : fbs_pos->memptr();
    dtype *p_lbs = lbs_pos == nullptr ? nullptr : lbs_pos->memptr();
    dtype *p_angles = path_angles == nullptr ? nullptr : path_angles->memptr();

    // Calculate half way point between TX and RX
    dtype TRx = Rx - Tx, TRy = Ry - Ty, TRz = Rz - Tz;
    TRx = Tx + half * TRx, TRy = Ty + half * TRy, TRz = Tz + half * TRz;

    for (size_t ip = 0; ip < n_path; ++ip)
    {
        dtype fx = TRx, fy = TRy, fz = TRz;     // Initial FBS-Pos = half way point
        dtype lx = TRx, ly = TRy, lz = TRz;     // Initial LBS-Pos = half way point
        dtype x = Tx, y = Ty, z = Tz, d = zero; // Set segment start to TX position

        dtype *ppc = nullptr;
        if (path_coord != nullptr)
        {
            path_coord->at(ip).set_size(3, p_interact[ip] + 2);
            ppc = path_coord->at(ip).memptr();
            *ppc++ = Tx, *ppc++ = Ty, *ppc++ = Tz;
        }

        // Get FBS and LBS positions
        for (unsigned ii = 0; ii < p_interact[ip]; ++ii)
        {
            lx = *p_coord++, ly = *p_coord++, lz = *p_coord++;                      // Read segment end coordinate
            x -= lx, y -= ly, z -= lz;                                              // Calculate vector pointing from segment start to segment end
            d += std::sqrt(x * x + y * y + z * z);                                  // Add segment length to total path length
            x = lx, y = ly, z = lz;                                                 // Update segment start for next segment
            fx = ii == 0 ? lx : fx, fy = ii == 0 ? ly : fy, fz = ii == 0 ? lz : fz; // Sore FBS position (segment 0)
            if (ppc != nullptr)
                *ppc++ = lx, *ppc++ = ly, *ppc++ = lz;
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
        if (ppc != nullptr)
            *ppc++ = Rx, *ppc++ = Ry, *ppc++ = Rz;
    }
}

template void quadriga_lib::coord2path(float Tx, float Ty, float Tz, float Rx, float Ry, float Rz, const arma::Col<unsigned> *no_interact, const arma::Mat<float> *interact_coord,
                                       arma::Col<float> *path_length, arma::Mat<float> *fbs_pos, arma::Mat<float> *lbs_pos, arma::Mat<float> *path_angles, std::vector<arma::Mat<float>> *path_coord);

template void quadriga_lib::coord2path(double Tx, double Ty, double Tz, double Rx, double Ry, double Rz, const arma::Col<unsigned> *no_interact, const arma::Mat<double> *interact_coord,
                                       arma::Col<double> *path_length, arma::Mat<double> *fbs_pos, arma::Mat<double> *lbs_pos, arma::Mat<double> *path_angles, std::vector<arma::Mat<double>> *path_coord);

// Construct a geodesic polyhedron (icosphere), a convex polyhedron made from triangles
template <typename dtype>
size_t quadriga_lib::icosphere(arma::uword n_div, dtype radius, arma::Mat<dtype> *center, arma::Col<dtype> *length,
                               arma::Mat<dtype> *vert, arma::Mat<dtype> *direction, bool direction_xyz)
{
    if (n_div == 0)
        throw std::invalid_argument("Input 'n_div' cannot be 0.");

    if (radius < dtype(0.0))
        throw std::invalid_argument("Input 'radius' cannot be negative.");

    arma::uword n_faces = n_div * n_div * 20;
    size_t n_faces_t = (size_t)n_faces;

    // Set sizes
    if (center == nullptr)
        throw std::invalid_argument("Output 'center' cannot be NULL.");

    if (center->n_rows != n_faces || center->n_cols != 3)
        center->set_size(n_faces, 3);

    if (length != nullptr && length->n_elem != n_faces)
        length->set_size(n_faces);

    if (vert != nullptr && (vert->n_rows != n_faces || vert->n_cols != 9))
        vert->set_size(n_faces, 9);

    int calc_directions = (direction == nullptr) ? 0 : (direction_xyz ? 2 : 1);
    if (calc_directions == 1 && (direction->n_rows != n_faces || direction->n_cols != 6))
        direction->set_size(n_faces, 6);
    else if (calc_directions == 2 && (direction->n_rows != n_faces || direction->n_cols != 9))
        direction->set_size(n_faces, 9);

    // Vertex coordinates of a regular isohedron
    double r = (double)radius, p = 1.618033988749895 * r, z = 0.0;
    double val[180] = {z, z, z, z, z, z, z, z, -p, -p, p, p, -r, -r, r, z, z, z, z, z,
                       r, r, r, r, r, -r, -r, -r, z, z, z, z, -p, -p, -p, r, r, r, r, r,
                       p, p, p, p, p, p, p, p, r, r, r, r, z, z, z, -p, -p, -p, -p, -p,
                       z, -p, p, -r, r, r, -r, -p, -r, -p, r, p, -p, z, z, -r, -p, z, p, r,
                       -r, z, z, p, p, -p, -p, z, p, z, -p, z, z, -r, -r, p, z, -r, z, p,
                       p, r, r, z, z, z, z, r, z, -r, z, -r, -r, -p, -p, z, -r, -p, -r, z,
                       p, z, r, -p, -r, p, r, -r, -p, -r, p, r, z, r, p, r, -r, -p, z, p,
                       z, -r, p, z, p, z, -p, -p, z, -p, z, p, -r, -p, z, p, p, z, -r, z,
                       r, p, z, r, z, r, z, z, -r, z, -r, z, -p, z, -r, z, z, -r, -p, -r};

    // Rotate x and y-coordinates slightly to avoid artifacts in regular grids
    constexpr double si = 0.0078329; // ~ 0.45 degree
    constexpr double co = 0.999969322368236;
    for (size_t n = 0; n < 20; ++n)
    {
        double tmp = val[n];
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
    arma::Mat<double> isohedron = arma::Mat<double>(val, 20, 9, false, true);

    // Subdivide faces of the isohedron
    arma::Mat<double> icosphere;
    quadriga_lib::subdivide_triangles(n_div, &isohedron, &icosphere);

    // Get pointers for direct memory access
    double *p_icosphere = icosphere.memptr();
    dtype *p_dest = center->memptr();
    dtype *p_length = (length == nullptr) ? nullptr : length->memptr();
    dtype *p_vert = (vert == nullptr) ? nullptr : vert->memptr();
    dtype *p_direction = calc_directions ? direction->memptr() : nullptr;

    // Process all faces
    double ri = 1.0 / r;
    for (size_t n = 0; n < n_faces_t; ++n)
    {
        // Project triangles onto the unit sphere
        // First vertex
        double tmp = r / std::sqrt(p_icosphere[n] * p_icosphere[n] +
                                   p_icosphere[n + n_faces_t] * p_icosphere[n + n_faces_t] +
                                   p_icosphere[n + 2 * n_faces_t] * p_icosphere[n + 2 * n_faces_t]);

        p_icosphere[n] *= tmp;
        p_icosphere[n + n_faces_t] *= tmp;
        p_icosphere[n + 2 * n_faces_t] *= tmp;

        // Second vertex
        tmp = r / std::sqrt(p_icosphere[n + 3 * n_faces_t] * p_icosphere[n + 3 * n_faces_t] +
                            p_icosphere[n + 4 * n_faces_t] * p_icosphere[n + 4 * n_faces_t] +
                            p_icosphere[n + 5 * n_faces_t] * p_icosphere[n + 5 * n_faces_t]);

        p_icosphere[n + 3 * n_faces_t] *= tmp;
        p_icosphere[n + 4 * n_faces_t] *= tmp;
        p_icosphere[n + 5 * n_faces_t] *= tmp;

        // Third vertex
        tmp = r / std::sqrt(p_icosphere[n + 6 * n_faces_t] * p_icosphere[n + 6 * n_faces_t] +
                            p_icosphere[n + 7 * n_faces_t] * p_icosphere[n + 7 * n_faces_t] +
                            p_icosphere[n + 8 * n_faces_t] * p_icosphere[n + 8 * n_faces_t]);

        p_icosphere[n + 6 * n_faces_t] *= tmp;
        p_icosphere[n + 7 * n_faces_t] *= tmp;
        p_icosphere[n + 8 * n_faces_t] *= tmp;

        if (calc_directions == 1) // Spherical
        {
            // First vertex
            tmp = p_icosphere[n + 2 * n_faces_t] * ri;
            tmp = (tmp > 1.0) ? 1.0 : (tmp < -1.0 ? -1.0 : tmp);
            p_direction[n] = (dtype)std::atan2(p_icosphere[n + n_faces_t], p_icosphere[n]);
            p_direction[n + n_faces_t] = (dtype)std::asin(tmp);

            // Second vertex
            tmp = p_icosphere[n + 5 * n_faces_t] * ri;
            tmp = (tmp > 1.0) ? 1.0 : (tmp < -1.0 ? -1.0 : tmp);
            p_direction[n + 2 * n_faces_t] = (dtype)std::atan2(p_icosphere[n + 4 * n_faces_t], p_icosphere[n + 3 * n_faces_t]);
            p_direction[n + 3 * n_faces_t] = (dtype)std::asin(tmp);

            // Third vertex
            tmp = p_icosphere[n + 8 * n_faces_t] * ri;
            tmp = (tmp > 1.0) ? 1.0 : (tmp < -1.0 ? -1.0 : tmp);
            p_direction[n + 4 * n_faces_t] = (dtype)std::atan2(p_icosphere[n + 7 * n_faces_t], p_icosphere[n + 6 * n_faces_t]);
            p_direction[n + 5 * n_faces_t] = (dtype)std::asin(tmp);
        }
        else if (calc_directions == 2) // Cartesian
        {
            for (size_t m = 0; m < 9; ++m)
                p_direction[n + m * n_faces_t] = dtype(p_icosphere[n + m * n_faces_t] * ri);
        }

        // Calculate normal vector of the plane that is formed by the 3 vertices
        double Ux = p_icosphere[n + 3 * n_faces_t] - p_icosphere[n],
               Uy = p_icosphere[n + 4 * n_faces_t] - p_icosphere[n + n_faces_t],
               Uz = p_icosphere[n + 5 * n_faces_t] - p_icosphere[n + 2 * n_faces_t];

        double Vx = p_icosphere[n + 6 * n_faces_t] - p_icosphere[n],
               Vy = p_icosphere[n + 7 * n_faces_t] - p_icosphere[n + n_faces_t],
               Vz = p_icosphere[n + 8 * n_faces_t] - p_icosphere[n + 2 * n_faces_t];

        double Nx = Uy * Vz - Uz * Vy, Ny = Uz * Vx - Ux * Vz, Nz = Ux * Vy - Uy * Vx;       // Cross Product
        tmp = 1.0 / std::sqrt(Nx * Nx + Ny * Ny + Nz * Nz), Nx *= tmp, Ny *= tmp, Nz *= tmp; // Normalize

        // Distance from origin to plane
        tmp = (p_icosphere[n] * Nx + p_icosphere[n + n_faces_t] * Ny + p_icosphere[n + 2 * n_faces_t] * Nz);

        // Calculate intersect coordinate
        p_dest[n] = dtype(tmp * Nx);
        p_dest[n + n_faces_t] = dtype(tmp * Ny);
        p_dest[n + 2 * n_faces_t] = dtype(tmp * Nz);

        if (p_length != nullptr)
            p_length[n] = (dtype)std::abs(tmp);

        // Calculate vectors pointing from the face center to the triangle vertices
        if (p_vert != nullptr)
        {
            p_vert[n] = dtype(p_icosphere[n] - (double)p_dest[n]);
            p_vert[n + n_faces_t] = dtype(p_icosphere[n + n_faces_t] - (double)p_dest[n + n_faces_t]);
            p_vert[n + 2 * n_faces_t] = dtype(p_icosphere[n + 2 * n_faces_t] - (double)p_dest[n + 2 * n_faces_t]);
            p_vert[n + 3 * n_faces_t] = dtype(p_icosphere[n + 3 * n_faces_t] - (double)p_dest[n]);
            p_vert[n + 4 * n_faces_t] = dtype(p_icosphere[n + 4 * n_faces_t] - (double)p_dest[n + n_faces_t]);
            p_vert[n + 5 * n_faces_t] = dtype(p_icosphere[n + 5 * n_faces_t] - (double)p_dest[n + 2 * n_faces_t]);
            p_vert[n + 6 * n_faces_t] = dtype(p_icosphere[n + 6 * n_faces_t] - (double)p_dest[n]);
            p_vert[n + 7 * n_faces_t] = dtype(p_icosphere[n + 7 * n_faces_t] - (double)p_dest[n + n_faces_t]);
            p_vert[n + 8 * n_faces_t] = dtype(p_icosphere[n + 8 * n_faces_t] - (double)p_dest[n + 2 * n_faces_t]);
        }
    }

    return n_faces_t;
}

template size_t quadriga_lib::icosphere(arma::uword n_div, float radius, arma::Mat<float> *center, arma::Col<float> *length,
                                        arma::Mat<float> *vert, arma::Mat<float> *direction, bool direction_as_spheric);

template size_t quadriga_lib::icosphere(arma::uword n_div, double radius, arma::Mat<double> *center, arma::Col<double> *length,
                                        arma::Mat<double> *vert, arma::Mat<double> *direction, bool direction_as_spheric);

// Calculate the axis-aligned bounding box (AABB) of a 3D mesh
template <typename dtype>
arma::Mat<dtype> quadriga_lib::triangle_mesh_aabb(const arma::Mat<dtype> *mesh, const arma::Col<unsigned> *sub_mesh_index, size_t vec_size)
{
    // Input validation
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    if (mesh->n_rows == 0)
        throw std::invalid_argument("Input 'mesh' must have at least one face.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    size_t n_face_t = (size_t)mesh->n_rows;
    size_t n_mesh_t = (size_t)mesh->n_elem;
    const dtype *p_mesh = mesh->memptr();

    const unsigned first_sub_mesh_ind = 0;
    const unsigned *p_sub = &first_sub_mesh_ind;
    size_t n_sub = 1;

    if (sub_mesh_index != nullptr && sub_mesh_index->n_elem > 0)
    {
        n_sub = (size_t)sub_mesh_index->n_elem;
        if (n_sub == 0)
            throw std::invalid_argument("Input 'sub_mesh_index' must have at least one element.");

        p_sub = sub_mesh_index->memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-mesh must start at index 0.");

        for (size_t i = 1; i < n_sub; ++i)
            if (p_sub[i] <= p_sub[i - 1])
                throw std::invalid_argument("Sub-mesh indices must be sorted in ascending order.");

        if (p_sub[n_sub - 1] >= (unsigned)n_face_t)
            throw std::invalid_argument("Sub-mesh indices cannot exceed number of faces.");
    }

    // Reserve memory for the output
    size_t n_out = (n_sub % vec_size == 0) ? n_sub : (n_sub / vec_size + 1) * vec_size;
    arma::Mat<dtype> output(n_out, 6); // Initialized to 0

    dtype *x_min = output.colptr(0), *x_max = output.colptr(1),
          *y_min = output.colptr(2), *y_max = output.colptr(3),
          *z_min = output.colptr(4), *z_max = output.colptr(5);

    for (size_t i = 0; i < n_sub; ++i)
    {
        x_min[i] = INFINITY, x_max[i] = -INFINITY,
        y_min[i] = INFINITY, y_max[i] = -INFINITY,
        z_min[i] = INFINITY, z_max[i] = -INFINITY;
    }

    size_t i_sub = 0, i_next = n_face_t;
    for (size_t i_mesh = 0; i_mesh < n_mesh_t; ++i_mesh)
    {
        dtype v = p_mesh[i_mesh];         // Mesh value
        size_t i_col = i_mesh / n_face_t; // Column index in mesh
        size_t i_row = i_mesh % n_face_t; // Row index in mesh

        if (i_row == 0)
        {
            i_sub = 0;
            i_next = (i_sub == n_sub - 1) ? n_face_t : (size_t)p_sub[i_sub + 1];
        }
        else if (i_row == i_next)
        {
            ++i_sub;
            i_next = (i_sub == n_sub - 1) ? n_face_t : (size_t)p_sub[i_sub + 1];
        }

        if (i_col % 3 == 0)
        {
            x_min[i_sub] = (v < x_min[i_sub]) ? v : x_min[i_sub];
            x_max[i_sub] = (v > x_max[i_sub]) ? v : x_max[i_sub];
        }
        else if (i_col % 3 == 1)
        {
            y_min[i_sub] = (v < y_min[i_sub]) ? v : y_min[i_sub];
            y_max[i_sub] = (v > y_max[i_sub]) ? v : y_max[i_sub];
        }
        else
        {
            z_min[i_sub] = (v < z_min[i_sub]) ? v : z_min[i_sub];
            z_max[i_sub] = (v > z_max[i_sub]) ? v : z_max[i_sub];
        }
    }

    return output;
}
template arma::Mat<float> quadriga_lib::triangle_mesh_aabb(const arma::Mat<float> *mesh, const arma::Col<unsigned> *sub_mesh_index, size_t vec_size);
template arma::Mat<double> quadriga_lib::triangle_mesh_aabb(const arma::Mat<double> *mesh, const arma::Col<unsigned> *sub_mesh_index, size_t vec_size);

// Reorganize mesh into smaller sub-meshes for faster processing
template <typename dtype>
size_t quadriga_lib::triangle_mesh_segmentation(const arma::Mat<dtype> *mesh, arma::Mat<dtype> *meshR,
                                                arma::Col<unsigned> *sub_mesh_index, size_t target_size, size_t vec_size,
                                                const arma::Mat<dtype> *mtl_prop, arma::Mat<dtype> *mtl_propR, arma::Col<unsigned> *mesh_index)
{
    // Input validation
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    if (mesh->n_rows == 0)
        throw std::invalid_argument("Input 'mesh' must have at least one face.");

    size_t n_mesh_t = (size_t)mesh->n_rows;

    if (meshR == nullptr)
        throw std::invalid_argument("Output 'meshR' cannot be NULL.");
    if (sub_mesh_index == nullptr)
        throw std::invalid_argument("Output 'sub_mesh_index' cannot be NULL.");

    if (target_size == 0)
        throw std::invalid_argument("Input 'target_size' cannot be 0.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    bool process_mtl_prop = (mtl_prop != nullptr) && (mtl_propR != nullptr) && (mtl_prop->n_elem != 0);

    if (process_mtl_prop)
    {
        if (mtl_prop->n_cols != 5)
            throw std::invalid_argument("Input 'mtl_prop' must have 5 columns.");

        if (mtl_prop->n_rows != mesh->n_rows)
            throw std::invalid_argument("Number of rows in 'mesh' and 'mtl_prop' dont match.");
    }

    // Create a vector of meshes
    std::vector<arma::Mat<dtype>> c; // Vector of sub-meshes

    // Add first mesh (creates a copy of the data)
    c.push_back(*mesh);

    // Create base index (0-based)
    std::vector<arma::Col<size_t>> face_ind; // Index list
    {
        arma::Col<size_t> base_index(mesh->n_rows, arma::fill::none);
        size_t *p = base_index.memptr();
        for (size_t i = 0; i < n_mesh_t; ++i)
            p[i] = (size_t)i;
        face_ind.push_back(std::move(base_index));
    }

    // Iterate through all elements
    for (auto sub_mesh_it = c.begin(); sub_mesh_it != c.end();)
    {
        size_t n_sub_faces = (size_t)(*sub_mesh_it).n_rows;
        if (n_sub_faces > target_size)
        {
            arma::Mat<dtype> meshA, meshB;
            arma::Col<int> split_ind;

            // Split the mesh on its longest axis
            int split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 0, &split_ind);

            // Check the split proportions, must have at least a 10 / 90 split
            float p = (split_success > 0) ? (float)meshA.n_rows / (float)n_sub_faces : 0.5f;
            split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

            // Attempt to split along another axis if failed for longest axis
            int first_test = std::abs(split_success);
            if (split_success < 0) // Failed condition
            {
                // Test second axis
                if (first_test == 2 || first_test == 3) // Test x-axis
                    split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 1, &split_ind);
                else // Test y-axis
                    split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 2, &split_ind);

                p = (split_success > 0) ? (float)meshA.n_rows / (float)n_sub_faces : 0.5f;
                split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

                // If we still failed, we test the third axis
                if ((first_test == 1 && split_success == -2) || (first_test == 2 && split_success == -1))
                    split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 3, &split_ind);
                else if (first_test == 3 && split_success == -1)
                    split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 2, &split_ind);

                p = (split_success > 0) ? (float)meshA.n_rows / (float)n_sub_faces : 0.5f;
                split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;
            }

            // Update the mesh data in memory
            int *ps = split_ind.memptr();
            if (split_success > 0)
            {
                // Split the index list
                size_t i_sub = sub_mesh_it - c.begin();      // Sub-mesh index
                auto face_ind_it = face_ind.begin() + i_sub; // Face index iterator
                size_t *pi = (*face_ind_it).memptr();        // Current face index list

                arma::Col<size_t> meshA_index(meshA.n_rows, arma::fill::none);
                arma::Col<size_t> meshB_index(meshB.n_rows, arma::fill::none);
                size_t *pA = meshA_index.memptr(); // New face index list of mesh A
                size_t *pB = meshB_index.memptr(); // New face index list of mesh B

                size_t iA = 0, iB = 0;
                for (size_t i_face = 0; i_face < n_sub_faces; ++i_face)
                {
                    if (ps[i_face] == 1)
                        pA[iA++] = pi[i_face];
                    else if (ps[i_face] == 2)
                        pB[iB++] = pi[i_face];
                }

                face_ind.erase(face_ind_it);
                face_ind.push_back(std::move(meshA_index));
                face_ind.push_back(std::move(meshB_index));

                // Update the vector of sub-meshes
                c.erase(sub_mesh_it);
                c.push_back(std::move(meshA));
                c.push_back(std::move(meshB));
                sub_mesh_it = c.begin();
            }
            else
                ++sub_mesh_it;
        }
        else
            ++sub_mesh_it;
    }

    // Get the sub-mesh indices
    size_t n_sub = c.size(), n_out = 0;
    sub_mesh_index->set_size(n_sub);
    unsigned *p_sub_ind = sub_mesh_index->memptr();

    for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        size_t n_sub_faces = c[i_sub].n_rows;
        size_t n_align = (n_sub_faces % vec_size == 0) ? n_sub_faces : (n_sub_faces / vec_size + 1) * vec_size;
        p_sub_ind[i_sub] = (unsigned)n_out;
        n_out += n_align;
    }

    // Assemble output
    meshR->set_size(n_out, 9);
    dtype *p_mesh_out = meshR->memptr();

    const dtype *p_mtl_in = process_mtl_prop ? mtl_prop->memptr() : nullptr;
    dtype *p_mtl_out = nullptr;
    if (process_mtl_prop)
    {
        mtl_propR->set_size(n_out, 5);
        p_mtl_out = mtl_propR->memptr();
    }

    unsigned *p_mesh_index = nullptr;
    if (mesh_index != nullptr)
    {
        mesh_index->zeros(n_out);
        p_mesh_index = mesh_index->memptr();
    }

    for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        size_t n_sub_faces = c[i_sub].n_rows;  // Number of faces in the sub-mesh
        dtype *p_sub_mesh = c[i_sub].memptr(); // Pointer to sub-mesh data
        size_t *pi = face_ind[i_sub].memptr(); // Face index list of current sub-mesh

        // Copy sub-mesh data columns by column
        for (size_t i_col = 0; i_col < 9; ++i_col)
        {
            size_t offset = i_col * n_out + (size_t)p_sub_ind[i_sub];
            std::memcpy(&p_mesh_out[offset], &p_sub_mesh[i_col * n_sub_faces], n_sub_faces * sizeof(dtype));
        }

        // Copy material data
        if (process_mtl_prop)
            for (size_t i_col = 0; i_col < 5; ++i_col)
            {
                size_t offset_out = i_col * n_out + (size_t)p_sub_ind[i_sub];
                size_t offset_in = i_col * n_mesh_t;
                for (size_t i_sub_face = 0; i_sub_face < n_sub_faces; ++i_sub_face)
                    p_mtl_out[offset_out + i_sub_face] = p_mtl_in[offset_in + pi[i_sub_face]];
            }

        // Write mesh index
        if (p_mesh_index != nullptr)
        {
            size_t offset = (size_t)p_sub_ind[i_sub];
            for (size_t i_sub_face = 0; i_sub_face < n_sub_faces; ++i_sub_face)
                p_mesh_index[offset + i_sub_face] = (unsigned)pi[i_sub_face] + 1;
        }

        // Add padding data
        if (n_sub_faces % vec_size != 0)
        {
            // Calculate bounding box of current sub-mesh
            arma::Mat<dtype> aabb = quadriga_lib::triangle_mesh_aabb(&c[i_sub]);
            dtype *p_box = aabb.memptr();

            dtype x = p_box[0] + (dtype)0.5 * (p_box[1] - p_box[0]);
            dtype y = p_box[2] + (dtype)0.5 * (p_box[3] - p_box[2]);
            dtype z = p_box[4] + (dtype)0.5 * (p_box[5] - p_box[4]);

            size_t i_start = (size_t)p_sub_ind[i_sub] + n_sub_faces;
            size_t i_end = (i_sub == n_sub - 1) ? n_out : (size_t)p_sub_ind[i_sub + 1];

            for (size_t i_col = 0; i_col < 9; ++i_col)
                for (size_t i_pad = i_start; i_pad < i_end; ++i_pad)
                {
                    size_t offset = i_col * n_out + i_pad;
                    if (i_col % 3 == 0)
                        p_mesh_out[offset] = x;
                    else if (i_col % 3 == 1)
                        p_mesh_out[offset] = y;
                    else
                        p_mesh_out[offset] = z;

                    if (process_mtl_prop && i_col == 0)
                        p_mtl_out[offset] = (dtype)1.0;
                    else if (process_mtl_prop && i_col < 5)
                        p_mtl_out[offset] = (dtype)0.0;
                }
        }
    }

    return n_sub;
}

template size_t quadriga_lib::triangle_mesh_segmentation(const arma::Mat<float> *mesh, arma::Mat<float> *meshR,
                                                         arma::Col<unsigned> *sub_mesh_index, size_t target_size, size_t vec_size,
                                                         const arma::Mat<float> *mtl_prop, arma::Mat<float> *mtl_propR, arma::Col<unsigned> *mesh_index);

template size_t quadriga_lib::triangle_mesh_segmentation(const arma::Mat<double> *mesh, arma::Mat<double> *meshR,
                                                         arma::Col<unsigned> *sub_mesh_index, size_t target_size, size_t vec_size,
                                                         const arma::Mat<double> *mtl_prop, arma::Mat<double> *mtl_propR, arma::Col<unsigned> *mesh_index);

// Split the mesh into two along a given axis of the Coordinate system
template <typename dtype>
int quadriga_lib::triangle_mesh_split(const arma::Mat<dtype> *mesh, arma::Mat<dtype> *meshA, arma::Mat<dtype> *meshB, int axis, arma::Col<int> *split_ind)
{
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (meshA == nullptr)
        throw std::invalid_argument("Output 'meshA' cannot be NULL.");
    if (meshB == nullptr)
        throw std::invalid_argument("Output 'meshB' cannot be NULL.");

    // Calculate bounding box
    arma::Mat<dtype> aabb = quadriga_lib::triangle_mesh_aabb(mesh);

    size_t n_face_t = (size_t)mesh->n_rows;
    size_t n_mesh_t = (size_t)mesh->n_elem;
    const dtype *p_mesh = mesh->memptr();

    // Find longest axis
    dtype x = aabb.at(0, 1) - aabb.at(0, 0);
    dtype y = aabb.at(0, 3) - aabb.at(0, 2);
    dtype z = aabb.at(0, 5) - aabb.at(0, 4);

    if (axis == 0)
    {
        if (z >= y && z >= x)
            axis = 3;
        else if (y >= x && y >= z)
            axis = 2;
        else
            axis = 1;
    }
    else if (axis != 1 && axis != 2 && axis != 3)
        throw std::invalid_argument("Input 'axis' must have values 0, 1, 2 or 3.");

    // Define bounding box A
    dtype x_max = (axis == 1) ? aabb.at(0, 0) + (dtype)0.5 * x : aabb.at(0, 1),
          y_max = (axis == 2) ? aabb.at(0, 2) + (dtype)0.5 * y : aabb.at(0, 3),
          z_max = (axis == 3) ? aabb.at(0, 4) + (dtype)0.5 * z : aabb.at(0, 5);

    // Determine all mesh elements that are outside box A
    bool *isB = new bool[n_face_t](); // Init to false

    for (size_t i_mesh = 0; i_mesh < n_mesh_t; ++i_mesh)
    {
        dtype v = p_mesh[i_mesh];         // Mesh value
        size_t i_col = i_mesh / n_face_t; // Column index in mesh
        size_t i_row = i_mesh % n_face_t; // Row index in mesh

        if (i_col % 3 == 0)
            isB[i_row] = (v > x_max) ? true : isB[i_row];
        else if (i_col % 3 == 1)
            isB[i_row] = (v > y_max) ? true : isB[i_row];
        else
            isB[i_row] = (v > z_max) ? true : isB[i_row];
    }

    // Count items in both sub-meshes
    size_t n_faceA = 0, n_faceB = 0;
    for (size_t i = 0; i < n_face_t; ++i)
        if (isB[i])
            ++n_faceB;
        else
            ++n_faceA;

    // Check if the mesh was split
    if (n_faceA == 0 || n_faceB == 0)
        return -axis;

    // Adjust output size
    meshA->set_size(n_faceA, 9);
    meshB->set_size(n_faceB, 9);

    dtype *p_meshA = meshA->memptr();
    dtype *p_meshB = meshB->memptr();

    bool write_split_ind = false;
    int *p_split_ind = nullptr;
    if (split_ind != nullptr)
    {
        write_split_ind = true;
        if (split_ind->n_elem != mesh->n_rows)
            split_ind->zeros(mesh->n_rows);
        else
            split_ind->zeros();
        p_split_ind = split_ind->memptr();
    }

    // Copy data
    size_t i_meshA = 0, i_meshB = 0;
    for (size_t i_mesh = 0; i_mesh < n_mesh_t; ++i_mesh)
    {
        dtype v = p_mesh[i_mesh];         // Mesh value
        size_t i_row = i_mesh % n_face_t; // Row index in mesh

        if (isB[i_row])
        {
            p_meshB[i_meshB++] = v;
            if (write_split_ind)
                p_split_ind[i_row] = 2;
        }
        else
        {
            p_meshA[i_meshA++] = v;
            if (write_split_ind)
                p_split_ind[i_row] = 1;
        }
    }

    delete[] isB;
    return axis;
}
template int quadriga_lib::triangle_mesh_split(const arma::Mat<float> *mesh, arma::Mat<float> *meshA, arma::Mat<float> *meshB, int axis, arma::Col<int> *split_ind);
template int quadriga_lib::triangle_mesh_split(const arma::Mat<double> *mesh, arma::Mat<double> *meshA, arma::Mat<double> *meshB, int axis, arma::Col<int> *split_ind);

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
size_t quadriga_lib::obj_file_read(std::string fn, arma::Mat<dtype> *mesh, arma::Mat<dtype> *mtl_prop, arma::Mat<dtype> *vert_list,
                                   arma::Mat<unsigned> *face_ind, arma::Col<unsigned> *obj_ind, arma::Col<unsigned> *mtl_ind,
                                   std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names)
{
    // Open file for reading
    std::ifstream fileR = std::ifstream(fn, std::ios::in);
    if (!fileR.is_open())
        throw std::invalid_argument("Error opening file.");

    // Obtain the number of faces and vertices from the file
    size_t n_vert = 0, n_faces = 0;
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
        size_t index;      // Material index
    };

    // Add default material data, See: Rec. ITU-R P.2040-1, Table 3
    std::vector<MaterialProp> mtl_lib;
    mtl_lib.push_back({"vacuum", 1.0, 0.0, 0.0, 0.0, 0.0, 0});
    mtl_lib.push_back({"air", 1.0, 0.0, 0.0, 0.0, 0.0, 0});
    mtl_lib.push_back({"textiles", 1.5, 0.0, 5e-5, 0.62, 0.0, 0});
    mtl_lib.push_back({"plastic", 2.44, 0.0, 2.33e-5, 1.0, 0.0, 0});
    mtl_lib.push_back({"ceramic", 6.5, 0.0, 0.0023, 1.32, 0.0, 0});
    mtl_lib.push_back({"sea_water", 80.0, -0.25, 4.0, 0.58, 0.0, 0});
    mtl_lib.push_back({"sea_ice", 3.2, -0.022, 1.1, 1.5, 0.0, 0});
    mtl_lib.push_back({"water", 80.0, -0.18, 0.6, 1.52, 0.0, 0});
    mtl_lib.push_back({"water_ice", 3.17, -0.005, 5.6e-5, 1.7, 0.0, 0});
    mtl_lib.push_back({"itu_concrete", 5.24, 0.0, 0.0462, 0.7822, 0.0, 0});
    mtl_lib.push_back({"itu_brick", 3.91, 0.0, 0.0238, 0.16, 0.0, 0});
    mtl_lib.push_back({"itu_plasterboard", 2.73, 0.0, 0.0085, 0.9395, 0.0, 0});
    mtl_lib.push_back({"itu_wood", 1.99, 0.0, 0.0047, 1.0718, 0.0, 0});
    mtl_lib.push_back({"itu_glass", 6.31, 0.0, 0.0036, 1.3394, 0.0, 0});
    mtl_lib.push_back({"itu_ceiling_board", 1.48, 0.0, 0.0011, 1.075, 0.0, 0});
    mtl_lib.push_back({"itu_chipboard", 2.58, 0.0, 0.0217, 0.78, 0.0, 0});
    mtl_lib.push_back({"itu_plywood", 2.71, 0.0, 0.33, 0.0, 0.0, 0});
    mtl_lib.push_back({"itu_marble", 7.074, 0.0, 0.0055, 0.9262, 0.0, 0});
    mtl_lib.push_back({"itu_floorboard", 3.66, 0.0, 0.0044, 1.3515, 0.0, 0});
    mtl_lib.push_back({"itu_metal", 1.0, 0.0, 1.0e7, 0.0, 0.0, 0});
    mtl_lib.push_back({"itu_very_dry_ground", 3.0, 0.0, 0.00015, 2.52, 0.0, 0});
    mtl_lib.push_back({"itu_medium_dry_ground", 15.0, -0.1, 0.035, 1.63, 0.0, 0});
    mtl_lib.push_back({"itu_wet_ground", 30.0, -0.4, 0.15, 1.3, 0.0, 0});
    mtl_lib.push_back({"itu_vegetation", 1.0, 0.0, 1.0e-4, 1.1, 0.0, 0}); // Rec. ITU-R P.833-9, Figure 2
    mtl_lib.push_back({"irr_glass", 6.27, 0.0, 0.0043, 1.1925, 23.0, 0}); // 3GPP TR 38.901 V17.0.0, Table 7.4.3-1: Material penetration losses

    // Reset the file pointer to the beginning of the file
    fileR.clear(); // Clear any flags
    fileR.seekg(0, std::ios::beg);

    // Local data
    size_t i_vert = 0, i_face = 0, j_face = 0, i_object = 0, i_mtl = 0; // Counters for vertices, faces, objects, materials
    size_t iM = 0;                                                      // Material index
    double aM = 1.0, bM = 0.0, cM = 0.0, dM = 0.0, attM = 0.0;          // Default material properties
    bool simple_face_format = true;                                     // Selector for face format

    // Obtain memory for the vertex list
    dtype *p_vert;
    if (vert_list == nullptr)
        p_vert = new dtype[n_vert * 3];
    else if (vert_list->n_rows != (arma::uword)n_vert || vert_list->n_cols != 3)
    {
        vert_list->set_size((arma::uword)n_vert, 3);
        p_vert = vert_list->memptr();
    }
    else
        p_vert = vert_list->memptr();

    // Obtain memory for face indices
    unsigned *p_face_ind;
    if (face_ind == nullptr)
        p_face_ind = new unsigned[n_faces * 3];
    else if (face_ind->n_rows != (arma::uword)n_faces || face_ind->n_cols != 3)
    {
        face_ind->set_size((arma::uword)n_faces, 3);
        p_face_ind = face_ind->memptr();
    }
    else
        p_face_ind = face_ind->memptr();

    // Set size of "mtl_prop"
    if (mtl_prop != nullptr && (mtl_prop->n_rows != (arma::uword)n_faces || mtl_prop->n_cols != 5))
        mtl_prop->set_size((arma::uword)n_faces, 5);
    dtype *p_mtl_prop = mtl_prop == nullptr ? nullptr : mtl_prop->memptr();

    // Set size of "mtl_ind"
    if (mtl_ind != nullptr && mtl_ind->n_elem != (arma::uword)n_faces)
        mtl_ind->set_size((arma::uword)n_faces);
    unsigned *p_mtl_ind = mtl_ind == nullptr ? nullptr : mtl_ind->memptr();

    // Set size of "obj_ind"
    if (obj_ind != nullptr && obj_ind->n_elem != (arma::uword)n_faces)
        obj_ind->set_size((arma::uword)n_faces);
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
            p_vert[i_vert++ + 2 * n_vert] = (dtype)z;
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
                p_mtl_ind[i_face] = (unsigned)iM;

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
                for (size_t i = j_face; i < i_face; ++i)
                    p_obj_ind[i] = (unsigned)i_object;

            // Add object name to list of object names
            if (obj_names != nullptr)
            {
                std::string obj_name = line.substr(2, 255); // Name in OBJ File
                obj_names->push_back(obj_name);
            }

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
                    mtl_lib.push_back({mtl_name, 1.0, 0.0, 0.0, 0.0, 0.0, 0}); // vacuum / air
                else
                    mtl_lib.push_back({mtl_name, aM, bM, cM, dM, attM, 0});
                found = (int)mtl_lib.size() - 1;
            }

            if (iM == 0) // Increase material counter
            {
                iM = ++i_mtl;
                mtl_lib[found].index = i_mtl;

                if (mtl_names != nullptr)
                    mtl_names->push_back(mtl_name);
            }
        }
    }

    // Set the object ID of the last object
    i_object = i_object == 0 ? 1 : i_object; // Single unnamed object
    if (p_obj_ind != nullptr)
        for (size_t i = j_face; i < i_face; ++i)
            p_obj_ind[i] = (unsigned)i_object;

    // Calculate the triangle mesh from vertices and faces
    if (mesh != nullptr)
    {
        if (mesh->n_rows != (arma::uword)n_faces || mesh->n_cols != 9)
            mesh->set_size((arma::uword)n_faces, 9);
        dtype *p_mesh = mesh->memptr();

        for (size_t n = 0; n < n_faces; ++n)
        {
            size_t a = p_face_ind[n],
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

template size_t quadriga_lib::obj_file_read(std::string fn, arma::Mat<float> *mesh, arma::Mat<float> *mtl_prop, arma::Mat<float> *vert_list,
                                            arma::Mat<unsigned> *face_ind, arma::Col<unsigned> *obj_ind, arma::Col<unsigned> *mtl_ind,
                                            std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names);

template size_t quadriga_lib::obj_file_read(std::string fn, arma::Mat<double> *mesh, arma::Mat<double> *mtl_prop, arma::Mat<double> *vert_list,
                                            arma::Mat<unsigned> *face_ind, arma::Col<unsigned> *obj_ind, arma::Col<unsigned> *mtl_ind,
                                            std::vector<std::string> *obj_names, std::vector<std::string> *mtl_names);

// Tests if 3D objects overlap (have a shared volume or boolean intersection)
template <typename dtype>
arma::u32_vec quadriga_lib::obj_overlap_test(const arma::Mat<dtype> *mesh, const arma::u32_vec *obj_ind,
                                             std::vector<std::string> *reason, dtype tolerance)
{
    const dtype eps = (dtype)1.0e-5; // Numeric tolerance

    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (obj_ind == nullptr)
        throw std::invalid_argument("Output 'obj_ind' cannot be NULL.");

    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");

    const arma::uword n_mesh = mesh->n_rows; // Number of mesh elements

    if (obj_ind->n_elem != n_mesh)
        throw std::invalid_argument("Number of elements in 'obj_ind' must match number of rows in 'mesh'.");

    // Get the unique object IDs
    arma::u32_vec obj_ids = arma::unique(*obj_ind); // Unique object IDs
    arma::uword n_obj = obj_ids.n_elem;             // Number of objects
    unsigned *p_obj_ids = obj_ids.memptr();         // Unique object IDs (pointer)
    const unsigned *p_obj_ind = obj_ind->memptr();  // Object IDs (pointer)

    arma::uvec n_faces_per_object(n_obj);
    arma::uword *p_faces_per_object = n_faces_per_object.memptr();

    arma::Mat<dtype> aabb(n_obj, 6ULL, arma::fill::value(INFINITY));
    aabb.col(1).fill(-INFINITY);
    aabb.col(3).fill(-INFINITY);
    aabb.col(5).fill(-INFINITY);

    const dtype *p_mesh = mesh->memptr();
    dtype *Xmin = aabb.colptr(0), *Xmax = aabb.colptr(1);
    dtype *Ymin = aabb.colptr(2), *Ymax = aabb.colptr(3);
    dtype *Zmin = aabb.colptr(4), *Zmax = aabb.colptr(5);

    // Get the bounding boxes for each object
    for (arma::uword i_mesh = 0ULL; i_mesh < n_mesh; ++i_mesh)
    {
        // Find object index
        arma::uword i_obj = 0ULL;
        for (arma::uword i = 0ULL; i < n_obj; ++i)
        {
            i_obj = i;
            if (p_obj_ids[i] == p_obj_ind[i_mesh])
                break;
        }

        ++p_faces_per_object[i_obj];

        dtype x = p_mesh[i_mesh],
              y = p_mesh[i_mesh + n_mesh],
              z = p_mesh[i_mesh + 2ULL * n_mesh];

        Xmin[i_obj] = (x < Xmin[i_obj]) ? x : Xmin[i_obj], Xmax[i_obj] = (x > Xmax[i_obj]) ? x : Xmax[i_obj];
        Ymin[i_obj] = (y < Ymin[i_obj]) ? y : Ymin[i_obj], Ymax[i_obj] = (y > Ymax[i_obj]) ? y : Ymax[i_obj];
        Zmin[i_obj] = (z < Zmin[i_obj]) ? z : Zmin[i_obj], Zmax[i_obj] = (z > Zmax[i_obj]) ? z : Zmax[i_obj];

        x = p_mesh[i_mesh + 3ULL * n_mesh],
        y = p_mesh[i_mesh + 4ULL * n_mesh],
        z = p_mesh[i_mesh + 5ULL * n_mesh];

        Xmin[i_obj] = (x < Xmin[i_obj]) ? x : Xmin[i_obj], Xmax[i_obj] = (x > Xmax[i_obj]) ? x : Xmax[i_obj];
        Ymin[i_obj] = (y < Ymin[i_obj]) ? y : Ymin[i_obj], Ymax[i_obj] = (y > Ymax[i_obj]) ? y : Ymax[i_obj];
        Zmin[i_obj] = (z < Zmin[i_obj]) ? z : Zmin[i_obj], Zmax[i_obj] = (z > Zmax[i_obj]) ? z : Zmax[i_obj];

        x = p_mesh[i_mesh + 6ULL * n_mesh],
        y = p_mesh[i_mesh + 7ULL * n_mesh],
        z = p_mesh[i_mesh + 8ULL * n_mesh];

        Xmin[i_obj] = (x < Xmin[i_obj]) ? x : Xmin[i_obj], Xmax[i_obj] = (x > Xmax[i_obj]) ? x : Xmax[i_obj];
        Ymin[i_obj] = (y < Ymin[i_obj]) ? y : Ymin[i_obj], Ymax[i_obj] = (y > Ymax[i_obj]) ? y : Ymax[i_obj];
        Zmin[i_obj] = (z < Zmin[i_obj]) ? z : Zmin[i_obj], Zmax[i_obj] = (z > Zmax[i_obj]) ? z : Zmax[i_obj];
    }

    // Split the mesh into individual objects
    std::vector<arma::Mat<dtype>> obj_faces(n_obj);   // Mesh split into individual objects
    std::vector<arma::Mat<dtype>> obj_normals(n_obj); // Normal vectors N,U,V
    std::vector<arma::Mat<dtype>> obj_edges_orig(n_obj);
    std::vector<arma::Mat<dtype>> obj_edges_dest(n_obj);

    for (arma::uword i_obj = 0ULL; i_obj < n_obj; ++i_obj)
    {
        arma::uword n_faces = p_faces_per_object[i_obj];
        arma::uword n_edges = 3ULL * n_faces;
        obj_faces[i_obj] = arma::Mat<dtype>(n_faces, 9ULL, arma::fill::none);
        obj_normals[i_obj] = arma::Mat<dtype>(n_faces, 9ULL, arma::fill::none);
        obj_edges_orig[i_obj] = arma::Mat<dtype>(n_edges, 3ULL, arma::fill::none);
        obj_edges_dest[i_obj] = arma::Mat<dtype>(n_edges, 3ULL, arma::fill::none);

        dtype *p_faces = obj_faces[i_obj].memptr();
        dtype *p_normals = obj_normals[i_obj].memptr();
        dtype *p_edges_orig = obj_edges_orig[i_obj].memptr();
        dtype *p_edges_dest = obj_edges_dest[i_obj].memptr();

        arma::uword i_face = 0ULL;
        unsigned obj_id = p_obj_ids[i_obj];

        for (arma::uword i_mesh = 0ULL; i_mesh < n_mesh; ++i_mesh)
            if (p_obj_ind[i_mesh] == obj_id)
            {
                // Vertex 1
                dtype x0 = p_mesh[i_mesh];
                dtype y0 = p_mesh[i_mesh + n_mesh];
                dtype z0 = p_mesh[i_mesh + 2ULL * n_mesh];

                p_faces[i_face] = x0;
                p_faces[i_face + n_faces] = y0;
                p_faces[i_face + 2ULL * n_faces] = z0;

                p_edges_orig[i_face] = x0;
                p_edges_orig[i_face + n_edges] = y0;
                p_edges_orig[i_face + 2ULL * n_edges] = z0;

                p_edges_dest[i_face + 2ULL * n_faces] = x0;
                p_edges_dest[i_face + 2ULL * n_faces + n_edges] = y0;
                p_edges_dest[i_face + 2ULL * n_faces + 2ULL * n_edges] = z0;

                // Vertex 2
                dtype x1 = p_mesh[i_mesh + 3ULL * n_mesh];
                dtype y1 = p_mesh[i_mesh + 4ULL * n_mesh];
                dtype z1 = p_mesh[i_mesh + 5ULL * n_mesh];

                p_faces[i_face + 3ULL * n_faces] = x1;
                p_faces[i_face + 4ULL * n_faces] = y1;
                p_faces[i_face + 5ULL * n_faces] = z1;

                p_edges_orig[i_face + n_faces] = x1;
                p_edges_orig[i_face + n_faces + n_edges] = y1;
                p_edges_orig[i_face + n_faces + 2ULL * n_edges] = z1;

                p_edges_dest[i_face] = x1;
                p_edges_dest[i_face + n_edges] = y1;
                p_edges_dest[i_face + 2ULL * n_edges] = z1;

                // Vertex 3
                dtype x2 = p_mesh[i_mesh + 6ULL * n_mesh];
                dtype y2 = p_mesh[i_mesh + 7ULL * n_mesh];
                dtype z2 = p_mesh[i_mesh + 8ULL * n_mesh];

                p_faces[i_face + 6ULL * n_faces] = x2;
                p_faces[i_face + 7ULL * n_faces] = y2;
                p_faces[i_face + 8ULL * n_faces] = z2;

                p_edges_orig[i_face + 2ULL * n_faces] = x2;
                p_edges_orig[i_face + 2ULL * n_faces + n_edges] = y2;
                p_edges_orig[i_face + 2ULL * n_faces + 2ULL * n_edges] = z2;

                p_edges_dest[i_face + n_faces] = x2;
                p_edges_dest[i_face + n_faces + n_edges] = y2;
                p_edges_dest[i_face + n_faces + 2ULL * n_edges] = z2;

                // Calculate Edges E1 and E2
                x1 -= x0, y1 -= y0, z1 -= z0;
                x2 -= x0, y2 -= y0, z2 -= z0;

                // Calculate normal vector N - store on X2
                crossp(x1, y1, z1, x2, y2, z2, &x2, &y2, &z2, true);
                p_normals[i_face] = x2;
                p_normals[i_face + n_faces] = y2;
                p_normals[i_face + 2ULL * n_faces] = z2;

                // P0 -> P1 (X1) as a direction for U
                dtype len = (dtype)1.0 / std::sqrt(x1 * x1 + y1 * y1 + z1 * z1);
                x1 *= len, y1 *= len, z1 *= len;
                p_normals[i_face + 3ULL * n_faces] = x1;
                p_normals[i_face + 4ULL * n_faces] = y1;
                p_normals[i_face + 5ULL * n_faces] = z1;

                // V = N x U, store result in X2
                crossp(x2, y2, z2, x1, y1, z1, &x2, &y2, &z2, true);
                p_normals[i_face + 6ULL * n_faces] = x2;
                p_normals[i_face + 7ULL * n_faces] = y2;
                p_normals[i_face + 8ULL * n_faces] = z2;

                ++i_face;
            }
    }

    // Reasons
    std::vector<std::string> why;
    bool want_to_know = reason != nullptr;
    if (want_to_know)
        why.resize(n_obj);

    // List of intersecing objects
    int *intersecting_objects = new int[n_obj]();

    // Pairwise tests
#pragma omp parallel for
    for (int i_obj_1_int = 0; i_obj_1_int < (int)n_obj - 1; ++i_obj_1_int)
    {
        arma::uword i_obj_1 = (arma::uword)i_obj_1_int;

        arma::uword n_faces_1 = p_faces_per_object[i_obj_1];
        dtype *p_obj_1 = obj_faces[i_obj_1].memptr();
        dtype ax_low = Xmin[i_obj_1] - eps, ax_high = Xmax[i_obj_1] + eps,
              ay_low = Ymin[i_obj_1] - eps, ay_high = Ymax[i_obj_1] + eps,
              az_low = Zmin[i_obj_1] - eps, az_high = Zmax[i_obj_1] + eps;

        for (arma::uword i_obj_2 = i_obj_1 + 1ULL; i_obj_2 < n_obj; ++i_obj_2)
        {
            // Skip if both objects are already marked as intersecting
            if (intersecting_objects[i_obj_1] && intersecting_objects[i_obj_2])
                continue;

            arma::uword n_faces_2 = p_faces_per_object[i_obj_2];
            dtype *p_obj_2 = obj_faces[i_obj_2].memptr();
            dtype bx_low = Xmin[i_obj_2] - eps, bx_high = Xmax[i_obj_2] + eps,
                  by_low = Ymin[i_obj_2] - eps, by_high = Ymax[i_obj_2] + eps,
                  bz_low = Zmin[i_obj_2] - eps, bz_high = Zmax[i_obj_2] + eps;

            // Test if bounding boxes do not overlap
            if (!(ax_high >= bx_low && ax_low <= bx_high &&
                  ay_high >= by_low && ay_low <= by_high &&
                  az_high >= bz_low && az_low <= bz_high))
                continue;

            // Test if all vertices are identical (duplicate objects)
            bool test_condition = n_faces_1 == n_faces_2;
            if (test_condition)
                for (arma::uword i = 0ULL; i < 9ULL * n_faces_1; ++i)
                    if (std::abs(p_obj_1[i] - p_obj_2[i]) > tolerance)
                    {
                        test_condition = false;
                        break;
                    }

            if (test_condition)
            {
                intersecting_objects[i_obj_1] = 1;
                intersecting_objects[i_obj_2] = 1;

                if (want_to_know)
                {
                    why[i_obj_1] = "Identical with OBJ-ID " + std::to_string(p_obj_ids[i_obj_2]);
                    why[i_obj_2] = "Identical with OBJ-ID " + std::to_string(p_obj_ids[i_obj_1]);
                }

                continue;
            }

            // Test if edges of OBJ-1 intersect with faces of OBJ-2 or edges of OBJ-2 intersect with faces of OBJ-1
            test_condition = false;
            std::string findings;
            bool nothing_to_report_so_far = true;
            for (bool i12 : {true, false})
            {
                // Skip second test if first test was already positive
                if (test_condition)
                    break;

                // Get pointers to data
                arma::uword n_faces = i12 ? n_faces_1 : n_faces_2;
                arma::uword n_edges = i12 ? 3ULL * n_faces_2 : 3ULL * n_faces_1;
                arma::Mat<dtype> *orig = i12 ? &obj_edges_orig[i_obj_2] : &obj_edges_orig[i_obj_1];
                arma::Mat<dtype> *dest = i12 ? &obj_edges_dest[i_obj_2] : &obj_edges_dest[i_obj_1];
                arma::Mat<dtype> *edge_normals = i12 ? &obj_normals[i_obj_2] : &obj_normals[i_obj_1];
                arma::Mat<dtype> *faces = i12 ? &obj_faces[i_obj_1] : &obj_faces[i_obj_2];
                arma::Mat<dtype> *face_normals = i12 ? &obj_normals[i_obj_1] : &obj_normals[i_obj_2];

                // Test if edges intersect with faces (3D test)
                arma::u32_vec hit; // Face hit indicator, 1-based, 0 = no hit
                quadriga_lib::ray_triangle_intersect<dtype>(orig, dest, faces, nullptr, nullptr, nullptr, &hit);

                // False positives:
                // - Edge lies in the face plane (and hits at a random place due to numeric instabilities)
                // - Edge starts in the face plane (and causes random hit)
                // - Edge ends in the face plane (and causes random hit)

                // Use the hit indicator to test for different intersect reasons
                // 0 - no intersection, 1 - 2D intersections, >1 3D intersections
                unsigned *p_hit = hit.memptr();
                for (arma::uword i_edge = 0ULL; i_edge < n_edges; ++i_edge)
                    if (p_hit[i_edge])
                        ++p_hit[i_edge];

                dtype *p_faces = faces->memptr();
                dtype *p_face_normals = face_normals->memptr();
                dtype *p_orig = orig->memptr();
                dtype *p_dest = dest->memptr();
                dtype *p_edge_normals = edge_normals->memptr();

                // Iterate through all faces of OBJ-1
                for (arma::uword i_face = 0ULL; i_face < n_faces; ++i_face)
                {
                    // Face vertices
                    dtype x0 = p_faces[i_face],
                          y0 = p_faces[i_face + n_faces],
                          z0 = p_faces[i_face + 2ULL * n_faces];

                    dtype x1 = p_faces[i_face + 3ULL * n_faces],
                          y1 = p_faces[i_face + 4ULL * n_faces],
                          z1 = p_faces[i_face + 5ULL * n_faces];

                    dtype x2 = p_faces[i_face + 6ULL * n_faces],
                          y2 = p_faces[i_face + 7ULL * n_faces],
                          z2 = p_faces[i_face + 8ULL * n_faces];

                    // Face normal vectors
                    dtype Nx = p_face_normals[i_face],
                          Ny = p_face_normals[i_face + n_faces],
                          Nz = p_face_normals[i_face + 2ULL * n_faces];

                    dtype Ux = p_face_normals[i_face + 3ULL * n_faces],
                          Uy = p_face_normals[i_face + 4ULL * n_faces],
                          Uz = p_face_normals[i_face + 5ULL * n_faces];

                    dtype Vx = p_face_normals[i_face + 6ULL * n_faces],
                          Vy = p_face_normals[i_face + 7ULL * n_faces],
                          Vz = p_face_normals[i_face + 8ULL * n_faces];

                    // 3D to 2D projection
                    struct point2D
                    {
                        dtype x;
                        dtype y;
                    };
                    auto project2D = [&](dtype px, dtype py, dtype pz) -> point2D
                    {
                        dtype dx = px - x0, dy = py - y0, dz = pz - z0;
                        point2D res;
                        res.x = dx * Ux + dy * Uy + dz * Uz; // coordinate along Ux
                        res.y = dx * Vx + dy * Vy + dz * Vz; // coordinate along Uy
                        return res;
                    };
                    auto P1 = project2D(x1, y1, z1);
                    auto P2 = project2D(x2, y2, z2);

                    // Iterate through all edges of OBJ-2
                    for (arma::uword i_edge = 0ULL; i_edge < n_edges; ++i_edge)
                    {
                        // Edge start and end points
                        dtype ox = p_orig[i_edge],
                              oy = p_orig[i_edge + n_edges],
                              oz = p_orig[i_edge + 2ULL * n_edges];

                        dtype dx = p_dest[i_edge],
                              dy = p_dest[i_edge + n_edges],
                              dz = p_dest[i_edge + 2ULL * n_edges];

                        // Distance of edge start and end points from the face plane
                        dtype dist_o = std::fabs(Nx * (ox - x0) + Ny * (oy - y0) + Nz * (oz - z0));
                        dtype dist_d = std::fabs(Nx * (dx - x0) + Ny * (dy - y0) + Nz * (dz - z0));

                        // Is one point on the plane and the other outside?
                        if ((dist_o < tolerance && dist_d > tolerance) || (dist_o > tolerance && dist_d < tolerance))
                        {
                            p_hit[i_edge] = 0U; // Clear potential 3D hit
                            continue;
                        }

                        // Are both edge points in the face plane?
                        if (dist_o < tolerance && dist_d < tolerance)
                        {
                            // Check if the normal vectors of the face and the edge point in the same direction
                            arma::uword n_faces_o2 = n_edges / 3ULL;
                            dtype Mx = p_edge_normals[i_edge % n_faces_o2],
                                  My = p_edge_normals[i_edge % n_faces_o2 + n_faces_o2],
                                  Mz = p_edge_normals[i_edge % n_faces_o2 + 2ULL * n_faces_o2];

                            dtype dtp = dotp(Mx, My, Mz, Nx, Ny, Nz);
                            if (std::fabs(dtp - (dtype)1.0) > (dtype)1e-5) // Different directions?
                            {
                                p_hit[i_edge] = 0U; // Clear potential 3D hit
                                continue;
                            }

                            // 3D to 2D projection
                            auto O = project2D(ox, oy, oz);
                            auto D = project2D(dx, dy, dz);

                            int edge_hit = line_triangle_intersect_2D(P1.x, P2.x, P2.y, O.x, O.y, D.x, D.y, tolerance);

                            if (edge_hit != 0)      // Did we have a 2D hit?
                                p_hit[i_edge] = 1U; // A 2D hit overwrites a 3D miss!
                            else                    // Both edge points are on the plane, but no 2D hit was found
                                p_hit[i_edge] = 0U; // Clear any detected 3D hit!

                            // Debugging output
                            // std::cout << "F:" << i_face << " {(" << x0 << "," << y0 << "," << z0 << ")"
                            //           << " (" << x1 << "," << y1 << "," << z1 << ")"
                            //           << " (" << x2 << "," << y2 << "," << z2 << ")} ::"
                            //           << " {(" << 0 << "," << 0 << ")"
                            //           << " (" << P1.x << "," << P1.y << ")"
                            //           << " (" << P2.x << "," << P2.y << ")} ;"
                            //           << " E:" << i_edge << "(" << i_edge % n_faces << ") {(" << ox << "," << oy << "," << oz << ")"
                            //           << " (" << dx << "," << dy << "," << dz << ")} ::"
                            //           << " {(" << O.x << "," << O.y << ")"
                            //           << " (" << D.x << "," << D.y << ")} ;"
                            //           << " H:" << edge_hit << std::endl;

                            if (nothing_to_report_so_far && edge_hit != 0)
                            {
                                nothing_to_report_so_far = false;
                                std::ostringstream oss;
                                if (i12)
                                    oss << "2D Intersect: OBJ-IDs (" << p_obj_ids[i_obj_1] << "," << p_obj_ids[i_obj_2] << ")";
                                else
                                    oss << "2D Intersect: OBJ-IDs (" << p_obj_ids[i_obj_2] << "," << p_obj_ids[i_obj_1] << ")";

                                oss << " @ F:" << i_face << " {(" << x0 << "," << y0 << "," << z0 << ")"
                                    << ",(" << x1 << "," << y1 << "," << z1 << ")"
                                    << ",(" << x2 << "," << y2 << "," << z2 << ")}";

                                oss << " vs. E:" << i_edge << "(" << i_edge % n_faces << ") {(" << ox << "," << oy << "," << oz << ")"
                                    << ",(" << dx << "," << dy << "," << dz << ")} => ";

                                if (edge_hit == -1)
                                    oss << "degenerate face (-1)";
                                else if (edge_hit == 1)
                                    oss << "co-planar edge starts or ends within face (1)";
                                else if (edge_hit == 2)
                                    oss << "co-planar edge inside face (2)";
                                else if (edge_hit == 4)
                                    oss << "co-planar vertices lie on face boundary (4)";
                                else if (edge_hit == 5)
                                    oss << "co-located vertices (5)";
                                else if (edge_hit == 6)
                                    oss << "co-planar edge passes through face (6)";
                                else if (edge_hit == 7)
                                    oss << "co-linear edges (7)";

                                findings = oss.str();
                            }
                        }
                    }
                }

                // Any hit?
                for (arma::uword i_edge = 0ULL; i_edge < n_edges; ++i_edge)
                    if (p_hit[i_edge])
                    {
                        test_condition = true;
                        if (nothing_to_report_so_far && p_hit[i_edge] > 1U)
                        {
                            nothing_to_report_so_far = false;
                            std::ostringstream oss;
                            if (i12)
                                oss << "3D Intersect: OBJ-IDs (" << p_obj_ids[i_obj_1] << "," << p_obj_ids[i_obj_2] << ")";
                            else
                                oss << "3D Intersect: OBJ-IDs (" << p_obj_ids[i_obj_2] << "," << p_obj_ids[i_obj_1] << ")";

                            unsigned i_face = p_hit[i_edge] - 2U;
                            dtype x0 = p_faces[i_face],
                                  y0 = p_faces[i_face + n_faces],
                                  z0 = p_faces[i_face + 2ULL * n_faces];
                            dtype x1 = p_faces[i_face + 3ULL * n_faces],
                                  y1 = p_faces[i_face + 4ULL * n_faces],
                                  z1 = p_faces[i_face + 5ULL * n_faces];
                            dtype x2 = p_faces[i_face + 6ULL * n_faces],
                                  y2 = p_faces[i_face + 7ULL * n_faces],
                                  z2 = p_faces[i_face + 8ULL * n_faces];

                            oss << " @ F:" << i_face << " {(" << x0 << "," << y0 << "," << z0 << ")"
                                << ",(" << x1 << "," << y1 << "," << z1 << ")"
                                << ",(" << x2 << "," << y2 << "," << z2 << ")}";

                            dtype ox = p_orig[i_edge],
                                  oy = p_orig[i_edge + n_edges],
                                  oz = p_orig[i_edge + 2ULL * n_edges];
                            dtype dx = p_dest[i_edge],
                                  dy = p_dest[i_edge + n_edges],
                                  dz = p_dest[i_edge + 2ULL * n_edges];

                            oss << " vs. E:" << i_edge << "(" << i_edge % n_faces << ") {(" << ox << "," << oy << "," << oz << ")"
                                << ",(" << dx << "," << dy << "," << dz << ")}";

                            findings = oss.str();
                        }
                        if (test_condition && !nothing_to_report_so_far)
                            break;
                    }
            }

            if (test_condition)
            {
                intersecting_objects[i_obj_1] = 2;
                intersecting_objects[i_obj_2] = 2;

                if (want_to_know)
                {
                    why[i_obj_1] = findings;
                    why[i_obj_2] = findings;
                }

                continue;
            }
        }
    }

    // Get the number of intersecting objects
    arma::uword n_intersecting = 0ULL;
    for (arma::uword i_obj = 0ULL; i_obj < n_obj; ++i_obj)
        if (intersecting_objects[i_obj])
            ++n_intersecting;

    if (want_to_know)
        reason->resize(n_intersecting);

    // Return value: list of intersecting objects
    arma::u32_vec obj_ids_intersecting(n_intersecting);
    n_intersecting = 0ULL;
    for (arma::uword i_obj = 0ULL; i_obj < n_obj; ++i_obj)
        if (intersecting_objects[i_obj])
        {
            obj_ids_intersecting(n_intersecting) = p_obj_ids[i_obj];
            if (want_to_know)
                reason->at(n_intersecting) = why[i_obj];
            ++n_intersecting;
        }

    delete[] intersecting_objects;
    return obj_ids_intersecting;
}

template arma::u32_vec quadriga_lib::obj_overlap_test(const arma::Mat<float> *mesh, const arma::u32_vec *obj_ind, std::vector<std::string> *reason, float tolerance);
template arma::u32_vec quadriga_lib::obj_overlap_test(const arma::Mat<double> *mesh, const arma::u32_vec *obj_ind, std::vector<std::string> *reason, double tolerance);

// Convert paths to tubes
template <typename dtype>
void quadriga_lib::path_to_tube(const arma::Mat<dtype> *path_coord, arma::Mat<dtype> *vert, arma::umat *faces, dtype radius, size_t n_edges)
{
    if (path_coord == nullptr)
        throw std::invalid_argument("Input 'path_coord' cannot be NULL.");
    if (vert == nullptr)
        throw std::invalid_argument("Output 'vert' cannot be NULL.");
    if (faces == nullptr)
        throw std::invalid_argument("Output 'faces' cannot be NULL.");

    if (path_coord->n_rows != 3ULL)
        throw std::invalid_argument("Input 'path_coord' must have 3 rows containing x,y,z coordinates.");

    if (path_coord->n_cols < 2ULL)
        throw std::invalid_argument("Input 'path_coord' must have 2 or more columns.");

    double radius_d = (double)radius;
    if (radius_d <= 0.0)
        throw std::invalid_argument("Radius mut be larger than 0.");

    if (n_edges < 3ULL)
        throw std::invalid_argument("Number of edges mut be >= 3.");

    size_t n_coord = path_coord->n_cols;
    size_t n_segments = n_coord - 1ULL;

    const dtype *p_coord = path_coord->memptr();

    // At steep angles between path segments, the path is split and an additional ring of vertices is added
    // Wee need to determine the number of splits
    dtype angle_limit = (dtype)0.939692620785908; // cosd(10)
    size_t n_split = 0ULL;
    arma::uvec subseg_indices(n_segments);
    arma::uword *i_subseg = subseg_indices.memptr();
    for (size_t i_seg = 1ULL; i_seg < n_segments; ++i_seg)
    {
        // Read "v0"
        dtype x0 = p_coord[3ULL * (i_seg - 1ULL)];
        dtype y0 = p_coord[3ULL * (i_seg - 1ULL) + 1ULL];
        dtype z0 = p_coord[3ULL * (i_seg - 1ULL) + 2ULL];

        // Read "v1"
        dtype x1 = p_coord[3ULL * i_seg];
        dtype y1 = p_coord[3ULL * i_seg + 1ULL];
        dtype z1 = p_coord[3ULL * i_seg + 2ULL];

        // Read "v2"
        dtype x2 = p_coord[3ULL * (i_seg + 1ULL)];
        dtype y2 = p_coord[3ULL * (i_seg + 1ULL) + 1ULL];
        dtype z2 = p_coord[3ULL * (i_seg + 1ULL) + 2ULL];

        // Calculate the vectors "d" and "f"
        dtype dx = x0 - x1, dy = y0 - y1, dz = z0 - z1; // From v1 to v0
        dtype fx = x2 - x1, fy = y2 - y1, fz = z2 - z1; // From v1 to v2

        // Calculate the angle between vectors "d" and "f"
        dtype cos_ang_df = dotp(dx, dy, dz, fx, fy, fz, true);

        if (cos_ang_df > angle_limit)
            ++n_split;
        i_subseg[i_seg] = n_split;
    }

    size_t n_vert = (n_coord + n_split) * n_edges;
    size_t n_faces = (n_coord - 1ULL) * n_edges;

    if (vert->n_rows != 3ULL || vert->n_cols != n_vert)
        vert->zeros(3ULL, n_vert);

    if (faces->n_rows != 4 || faces->n_cols != n_faces)
        faces->zeros(4ULL, n_faces);

    dtype *p_vert = vert->memptr();
    arma::uword *p_face = faces->memptr();

    double x0 = 0.0, x1 = 0.0, x2 = 0.0;              // Point v0
    double y0 = 0.0, y1 = 0.0, y2 = 0.0;              // Point v1
    double z0 = 0.0, z1 = 0.0, z2 = 0.0;              // Point v2
    double dx = 0.0, dy = 0.0, dz = 0.0, len_d = 0.0; // Vector from v0 to v1
    double fx = 0.0, fy = 0.0, fz = 0.0, len_f = 0.0; // Vector from v1 to v2

    // Vector "u" is orthogonal to "d" and points to the first edge
    double ux = NAN, uy = NAN, uz = NAN;

    // Temporary storage for the vertices
    arma::mat orig(n_edges, 3ULL), dest(n_edges, 3ULL), fbs(n_edges, 3ULL);
    double *p_orig = orig.memptr();
    double *p_dest = dest.memptr();
    double *p_fbs = fbs.memptr();

    size_t i_vert = 0ULL; // Vertex counter
    for (size_t i_seg = 0ULL; i_seg < n_segments; ++i_seg)
    {
        // Vector "g" is orthogonal to "-d" and "f"
        double gx = NAN, gy = NAN, gz = NAN;

        // Vector "h" lies in between "-d" and "f"
        double hx = NAN, hy = NAN, hz = NAN;

        // The angle between vectors "-d" and "f"
        double ang_df = NAN;

        // Get start and end point of the current segment
        if (i_seg == 0ULL)
        {
            // Read "v0"
            x0 = (double)p_coord[3ULL * i_seg];
            y0 = (double)p_coord[3ULL * i_seg + 1ULL];
            z0 = (double)p_coord[3ULL * i_seg + 2ULL];

            // Read "v1"
            x1 = (double)p_coord[3ULL * (i_seg + 1ULL)];
            y1 = (double)p_coord[3ULL * (i_seg + 1ULL) + 1ULL];
            z1 = (double)p_coord[3ULL * (i_seg + 1ULL) + 2ULL];

            // Calculate "d"
            dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
            len_d = std::sqrt(dx * dx + dy * dy + dz * dz);
            double scl = 1.0 / len_d;
            dx *= scl, dy *= scl, dz *= scl;

            // Pick an arbitrary vector "u" that is orthogonal to "d"
            if (std::abs(dx) > 1.0e-6 || std::abs(dy) > 1.0e-6)
                ux = -dy, uy = dx, uz = 0.0;
            else
                ux = 0.0, uy = 1.0, uz = 0.0;
            scl = 1.0 / std::sqrt(ux * ux + uy * uy);
            ux *= scl, uy *= scl;
        }
        else // i_seg > 0
        {
            // Copy "v0", "v1" and "d" from the previous iteration
            x0 = x1, y0 = y1, z0 = z1;
            x1 = x2, y1 = y2, z1 = z2;
            dx = fx, dy = fy, dz = fz, len_d = len_f;
        }

        if (i_seg < n_segments - 1ULL)
        {
            // Read "v2"
            size_t ind = 3ULL * (i_seg + 2ULL);
            x2 = (double)p_coord[ind];
            y2 = (double)p_coord[ind + 1ULL];
            z2 = (double)p_coord[ind + 2ULL];

            // Calculate "f"
            fx = x2 - x1, fy = y2 - y1, fz = z2 - z1;
            len_f = std::sqrt(fx * fx + fy * fy + fz * fz);
            double scl = 1.0 / len_f;
            fx *= scl, fy *= scl, fz *= scl;

            // Calculate a normal-vector "g" that is orthogonal to "-d" and "f"
            crossp(-dx, -dy, -dz, fx, fy, fz, &gx, &gy, &gz, true);

            // Calculate the angle between vectors "-d" and "f"
            ang_df = dotp(-dx, -dy, -dz, fx, fy, fz, true);
            ang_df = std::acos(ang_df);

            // Calculate vector "h" that lies in between "-d" and "f"
            rotate_vector_around_axis(-dx, -dy, -dz, gx, gy, gz, 0.5 * ang_df, &hx, &hy, &hz, true);
        }

        // Generate origin points for the edges by rotating "u" around "d"
        for (size_t i_edge = 0ULL; i_edge < n_edges; ++i_edge)
        {
            double ex = ux, ey = uy, ez = uz;
            if (i_edge != 0ULL)
            {
                double theta = (double)i_edge * 6.283185307179586 / (double)n_edges;
                rotate_vector_around_axis(ux, uy, uz, dx, dy, dz, theta, &ex, &ey, &ez, true);
            }

            // Scale by radius
            ex = ex * radius_d;
            ey = ey * radius_d;
            ez = ez * radius_d;

            // Update FBS by shifting to v1
            if (i_seg == n_segments - 1ULL || i_subseg[i_seg] != i_subseg[i_seg + 1ULL])
            {
                p_fbs[i_edge] = ex + x1;
                p_fbs[i_edge + n_edges] = ey + y1;
                p_fbs[i_edge + 2ULL * n_edges] = ez + z1;
            }

            // Shift to v0
            ex += x0, ey += y0, ez += z0;

            // Write edge vertices to ray origin
            p_orig[i_edge] = ex;
            p_orig[i_edge + n_edges] = ey;
            p_orig[i_edge + 2ULL * n_edges] = ez;

            // For the first segment of a new sub-segment, these origin points ara also the starting points of the tube
            if (i_seg == 0ULL || i_subseg[i_seg - 1ULL] != i_subseg[i_seg])
            {
                p_vert[i_vert] = (dtype)ex;
                p_vert[i_vert + 1ULL] = (dtype)ey;
                p_vert[i_vert + 2ULL] = (dtype)ez;
                i_vert += 3ULL;
            }
        }

        // Projection mode
        if (i_seg < n_segments - 1ULL && i_subseg[i_seg] == i_subseg[i_seg + 1ULL])
        {
            // Generate destination points for the projection
            // - Minimum angle "ang_df" = 10 deg --> tand(80°) = 5.7
            double d = len_d + 5.7 * radius_d;
            for (size_t i_edge = 0ULL; i_edge < n_edges; ++i_edge)
            {
                p_dest[i_edge] = p_orig[i_edge] + d * dx;
                p_dest[i_edge + n_edges] = p_orig[i_edge + n_edges] + d * dy;
                p_dest[i_edge + 2ULL * n_edges] = p_orig[i_edge + 2ULL * n_edges] + d * dz;
            }

            // Build a projection plane (triangle) to calculate the destination points of the tube
            // - Triangle has an inner radius of "0.5 * radius"
            // - When hitting at an incident angle of 10°, we need to scale by at lest 0.5 / sin(10°) = 2.9
            d = 20.0 * radius_d;
            arma::mat proj_plane = {d * gx + x1, d * gy + y1, d * gz + z1,
                                    d * (hx - gx) + x1, d * (hy - gy) + y1, d * (hz - gz) + z1,
                                    d * (-hx - gx) + x1, d * (-hy - gy) + y1, d * (-hz - gz) + z1};

            // Calculate intersection points with the plane
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &proj_plane, &fbs);
        }

        // Write FBS coordinates to output
        for (size_t i_edge = 0ULL; i_edge < n_edges; ++i_edge)
        {
            p_vert[i_vert] = (dtype)p_fbs[i_edge];
            p_vert[i_vert + 1ULL] = (dtype)p_fbs[i_edge + n_edges];
            p_vert[i_vert + 2ULL] = (dtype)p_fbs[i_edge + 2ULL * n_edges];
            i_vert += 3ULL;
        }

        // Update "u" for the next segment by rotating it around "-g"
        ang_df = dotp(dx, dy, dz, fx, fy, fz, true);
        ang_df = std::acos(ang_df);
        rotate_vector_around_axis(ux, uy, uz, -gx, -gy, -gz, ang_df, &ux, &uy, &uz, true);

        // Build face matrix
        for (size_t i_edge = 0ULL; i_edge < n_edges; ++i_edge)
        {
            size_t ind = 4ULL * (n_edges * i_seg + i_edge);
            size_t offset = i_subseg[i_seg] * n_edges;
            p_face[ind] = n_edges * i_seg + offset + i_edge;
            p_face[ind + 1ULL] = n_edges * i_seg + offset + (i_edge + 1ULL) % n_edges;
            p_face[ind + 2ULL] = n_edges * (i_seg + 1ULL) + offset + (i_edge + 1ULL) % n_edges;
            p_face[ind + 3ULL] = n_edges * (i_seg + 1ULL) + offset + i_edge;
        }
    }
}

template void quadriga_lib::path_to_tube(const arma::Mat<float> *path_coord, arma::Mat<float> *vert, arma::umat *faces,
                                         float radius, size_t n_edges);

template void quadriga_lib::path_to_tube(const arma::Mat<double> *path_coord, arma::Mat<double> *vert, arma::umat *faces,
                                         double radius, size_t n_edges);

// Calculate the axis-aligned bounding box (AABB) of a point cloud
template <typename dtype>
arma::Mat<dtype> quadriga_lib::point_cloud_aabb(const arma::Mat<dtype> *points, const arma::Col<unsigned> *sub_cloud_index, size_t vec_size)
{
    // Input validation
    if (points == nullptr)
        throw std::invalid_argument("Input 'points' cannot be NULL.");
    if (points->n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns containing x,y,z coordinates.");
    if (points->n_rows == 0)
        throw std::invalid_argument("Input 'points' must have at least one entry.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    size_t n_points_t = (size_t)points->n_rows;
    size_t n_values_t = (size_t)points->n_elem;
    const dtype *p_points = points->memptr();

    const unsigned first_sub_cloud_ind = 0;
    const unsigned *p_sub = &first_sub_cloud_ind;
    size_t n_sub = 1;

    if (sub_cloud_index != nullptr && sub_cloud_index->n_elem > 0)
    {
        n_sub = (size_t)sub_cloud_index->n_elem;
        if (n_sub == 0)
            throw std::invalid_argument("Input 'sub_cloud_index' must have at least one element.");

        p_sub = sub_cloud_index->memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-cloud must start at index 0.");

        for (size_t i = 1; i < n_sub; ++i)
            if (p_sub[i] <= p_sub[i - 1])
                throw std::invalid_argument("Sub-cloud indices must be sorted in ascending order.");

        if (p_sub[n_sub - 1] >= (unsigned)n_points_t)
            throw std::invalid_argument("Sub-cloud indices cannot exceed number of points.");
    }

    // Reserve memory for the output
    size_t n_out = (n_sub % vec_size == 0) ? n_sub : (n_sub / vec_size + 1) * vec_size;
    arma::Mat<dtype> output(n_out, 6); // Initialized to 0

    dtype *x_min = output.colptr(0), *x_max = output.colptr(1),
          *y_min = output.colptr(2), *y_max = output.colptr(3),
          *z_min = output.colptr(4), *z_max = output.colptr(5);

    for (size_t i = 0; i < n_sub; ++i)
    {
        x_min[i] = INFINITY, x_max[i] = -INFINITY,
        y_min[i] = INFINITY, y_max[i] = -INFINITY,
        z_min[i] = INFINITY, z_max[i] = -INFINITY;
    }

    size_t i_sub = 0, i_next = n_points_t;
    for (size_t i_point = 0; i_point < n_values_t; ++i_point)
    {
        dtype v = p_points[i_point];         // Point value
        size_t i_col = i_point / n_points_t; // Column index in mesh
        size_t i_row = i_point % n_points_t; // Row index in mesh

        if (i_row == 0)
        {
            i_sub = 0;
            i_next = (i_sub == n_sub - 1) ? n_points_t : (size_t)p_sub[i_sub + 1];
        }
        else if (i_row == i_next)
        {
            ++i_sub;
            i_next = (i_sub == n_sub - 1) ? n_points_t : (size_t)p_sub[i_sub + 1];
        }

        if (i_col == 0)
        {
            x_min[i_sub] = (v < x_min[i_sub]) ? v : x_min[i_sub];
            x_max[i_sub] = (v > x_max[i_sub]) ? v : x_max[i_sub];
        }
        else if (i_col == 1)
        {
            y_min[i_sub] = (v < y_min[i_sub]) ? v : y_min[i_sub];
            y_max[i_sub] = (v > y_max[i_sub]) ? v : y_max[i_sub];
        }
        else
        {
            z_min[i_sub] = (v < z_min[i_sub]) ? v : z_min[i_sub];
            z_max[i_sub] = (v > z_max[i_sub]) ? v : z_max[i_sub];
        }
    }

    return output;
}

template arma::Mat<float> quadriga_lib::point_cloud_aabb(const arma::Mat<float> *points, const arma::Col<unsigned> *sub_cloud_index, size_t vec_size);
template arma::Mat<double> quadriga_lib::point_cloud_aabb(const arma::Mat<double> *points, const arma::Col<unsigned> *sub_cloud_index, size_t vec_size);

// Reorganize a point cloud into smaller sub-clouds for faster processing
template <typename dtype>
size_t quadriga_lib::point_cloud_segmentation(const arma::Mat<dtype> *points, arma::Mat<dtype> *pointsR, arma::Col<unsigned> *sub_cloud_index,
                                              size_t target_size, size_t vec_size, arma::Col<unsigned> *forward_index, arma::Col<unsigned> *reverse_index)
{

    // Input validation
    if (points == nullptr)
        throw std::invalid_argument("Input 'points' cannot be NULL.");
    if (points->n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns containing x,y,z coordinates.");
    if (points->n_rows == 0)
        throw std::invalid_argument("Input 'points' must have at least one face.");

    size_t n_points_t = (size_t)points->n_rows;

    if (pointsR == nullptr)
        throw std::invalid_argument("Output 'pointsR' cannot be NULL.");
    if (sub_cloud_index == nullptr)
        throw std::invalid_argument("Output 'sub_cloud_index' cannot be NULL.");

    if (target_size == 0)
        throw std::invalid_argument("Input 'target_size' cannot be 0.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    // Create a vector of sub-clouds
    std::vector<arma::Mat<dtype>> c; // Vector of sub-clouds

    // Add first item (creates a copy of the data)
    c.push_back(*points);

    // Create base index (0-based)
    std::vector<arma::Col<size_t>> fwd_ind; // Index list
    {
        arma::Col<size_t> base_index(points->n_rows, arma::fill::none);
        size_t *p = base_index.memptr();
        for (size_t i = 0; i < n_points_t; ++i)
            p[i] = (size_t)i;
        fwd_ind.push_back(std::move(base_index));
    }

    // Iterate through all elements
    for (auto sub_cloud_it = c.begin(); sub_cloud_it != c.end();)
    {
        size_t n_sub_points = (size_t)(*sub_cloud_it).n_rows;
        if (n_sub_points > target_size)
        {
            arma::Mat<dtype> pointsA, pointsB;
            arma::Col<int> split_ind;

            // Split on longest axis
            int split_success = quadriga_lib::point_cloud_split(&(*sub_cloud_it), &pointsA, &pointsB, 0, &split_ind);

            // Check the split proportions, must have at least a 10 / 90 split
            float p = (split_success > 0) ? (float)pointsA.n_rows / (float)n_sub_points : 0.5f;
            split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

            // Attempt to split along another axis if failed for longest axis
            int first_test = std::abs(split_success);
            if (split_success < 0) // Failed condition
            {
                // Test second axis
                if (first_test == 2 || first_test == 3) // Test x-axis
                    split_success = quadriga_lib::point_cloud_split(&(*sub_cloud_it), &pointsA, &pointsB, 1, &split_ind);
                else // Test y-axis
                    split_success = quadriga_lib::point_cloud_split(&(*sub_cloud_it), &pointsA, &pointsB, 2, &split_ind);

                p = (split_success > 0) ? (float)pointsA.n_rows / (float)n_sub_points : 0.5f;
                split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

                // If we still failed, we test the third axis
                if ((first_test == 1 && split_success == -2) || (first_test == 2 && split_success == -1))
                    split_success = quadriga_lib::point_cloud_split(&(*sub_cloud_it), &pointsA, &pointsB, 3, &split_ind);
                else if (first_test == 3 && split_success == -1)
                    split_success = quadriga_lib::point_cloud_split(&(*sub_cloud_it), &pointsA, &pointsB, 2, &split_ind);

                p = (split_success > 0) ? (float)pointsA.n_rows / (float)n_sub_points : 0.5f;
                split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;
            }

            // Update the point cloud data in memory
            int *ps = split_ind.memptr();
            if (split_success > 0)
            {
                // Split the index list
                size_t i_sub = sub_cloud_it - c.begin();   // Sub-cloud index
                auto fwd_ind_it = fwd_ind.begin() + i_sub; // Forward index iterator
                size_t *pi = (*fwd_ind_it).memptr();       // Current cloud index list

                arma::Col<size_t> pointsA_index(pointsA.n_rows, arma::fill::none);
                arma::Col<size_t> pointsB_index(pointsB.n_rows, arma::fill::none);
                size_t *pA = pointsA_index.memptr(); // New face index list of mesh A
                size_t *pB = pointsB_index.memptr(); // New face index list of mesh B

                size_t iA = 0, iB = 0;
                for (size_t i_point = 0; i_point < n_sub_points; ++i_point)
                {
                    if (ps[i_point] == 1)
                        pA[iA++] = pi[i_point];
                    else if (ps[i_point] == 2)
                        pB[iB++] = pi[i_point];
                }

                fwd_ind.erase(fwd_ind_it);
                fwd_ind.push_back(std::move(pointsA_index));
                fwd_ind.push_back(std::move(pointsB_index));

                // Update the vector of sub-meshes
                c.erase(sub_cloud_it);
                c.push_back(std::move(pointsA));
                c.push_back(std::move(pointsB));
                sub_cloud_it = c.begin();
            }
            else
                ++sub_cloud_it;
        }
        else
            ++sub_cloud_it;
    }

    // Get the sub-cloud indices
    size_t n_sub = c.size(), n_out = 0;
    sub_cloud_index->set_size(n_sub);
    unsigned *p_sub_ind = sub_cloud_index->memptr();

    for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        size_t n_sub_points = c[i_sub].n_rows;
        size_t n_align = (n_sub_points % vec_size == 0) ? n_sub_points : (n_sub_points / vec_size + 1) * vec_size;
        p_sub_ind[i_sub] = (unsigned)n_out;
        n_out += n_align;
    }

    // Assemble output
    pointsR->set_size(n_out, 3);
    dtype *p_points_out = pointsR->memptr();

    unsigned *p_forward_index = nullptr;
    if (forward_index != nullptr || reverse_index != nullptr)
        p_forward_index = new unsigned[n_out]();

    for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        size_t n_sub_points = c[i_sub].n_rows;  // Number of points in the sub-cloud
        dtype *p_sub_cloud = c[i_sub].memptr(); // Pointer to sub-cloud data
        size_t *pi = fwd_ind[i_sub].memptr();   // Point index list of current sub-cloud

        // Copy sub-cloud data columns by column
        for (size_t i_col = 0; i_col < 3; ++i_col)
        {
            size_t offset = i_col * n_out + (size_t)p_sub_ind[i_sub];
            std::memcpy(&p_points_out[offset], &p_sub_cloud[i_col * n_sub_points], n_sub_points * sizeof(dtype));
        }

        // Write index
        if (p_forward_index != nullptr)
        {
            size_t offset = (size_t)p_sub_ind[i_sub];
            for (size_t i_sub_point = 0; i_sub_point < n_sub_points; ++i_sub_point)
                p_forward_index[offset + i_sub_point] = (unsigned)pi[i_sub_point] + 1;
        }

        // Add padding data
        if (n_sub_points % vec_size != 0)
        {
            // Calculate bounding box of current sub-mesh
            arma::Mat<dtype> aabb = quadriga_lib::point_cloud_aabb(&c[i_sub]);
            dtype *p_box = aabb.memptr();

            dtype x = p_box[0] + (dtype)0.5 * (p_box[1] - p_box[0]);
            dtype y = p_box[2] + (dtype)0.5 * (p_box[3] - p_box[2]);
            dtype z = p_box[4] + (dtype)0.5 * (p_box[5] - p_box[4]);

            size_t i_start = (size_t)p_sub_ind[i_sub] + n_sub_points;
            size_t i_end = (i_sub == n_sub - 1) ? n_out : (size_t)p_sub_ind[i_sub + 1];

            for (size_t i_col = 0; i_col < 3; ++i_col)
                for (size_t i_pad = i_start; i_pad < i_end; ++i_pad)
                {
                    size_t offset = i_col * n_out + i_pad;
                    if (i_col == 0)
                        p_points_out[offset] = x;
                    else if (i_col == 1)
                        p_points_out[offset] = y;
                    else
                        p_points_out[offset] = z;
                }
        }
    }

    // Copy forward index
    if (forward_index != nullptr)
    {
        forward_index->set_size(n_out);
        std::memcpy(forward_index->memptr(), p_forward_index, n_out * sizeof(unsigned));
    }

    // Generate reverse index
    if (reverse_index != nullptr)
    {
        reverse_index->set_size(n_points_t);
        unsigned *p = reverse_index->memptr();
        for (unsigned i = 0; i < n_out; ++i)
            if (p_forward_index[i] != 0)
                p[p_forward_index[i] - 1] = i;
    }

    if (forward_index != nullptr || reverse_index != nullptr)
        delete[] p_forward_index;

    return n_sub;
}

template size_t quadriga_lib::point_cloud_segmentation(const arma::Mat<float> *points, arma::Mat<float> *pointsR, arma::Col<unsigned> *sub_cloud_index,
                                                       size_t target_size, size_t vec_size, arma::Col<unsigned> *forward_index, arma::Col<unsigned> *reverse_index);

template size_t quadriga_lib::point_cloud_segmentation(const arma::Mat<double> *points, arma::Mat<double> *pointsR, arma::Col<unsigned> *sub_cloud_index,
                                                       size_t target_size, size_t vec_size, arma::Col<unsigned> *forward_index, arma::Col<unsigned> *reverse_index);

// Split a point cloud into two sub-clouds along a given axis
template <typename dtype>
int quadriga_lib::point_cloud_split(const arma::Mat<dtype> *points, arma::Mat<dtype> *pointsA, arma::Mat<dtype> *pointsB, int axis, arma::Col<int> *split_ind)
{
    if (points == nullptr)
        throw std::invalid_argument("Input 'points' cannot be NULL.");
    if (pointsA == nullptr)
        throw std::invalid_argument("Output 'pointsA' cannot be NULL.");
    if (pointsB == nullptr)
        throw std::invalid_argument("Output 'pointsB' cannot be NULL.");

    // Calculate bounding box
    arma::Mat<dtype> aabb = quadriga_lib::point_cloud_aabb(points);

    size_t n_points_t = (size_t)points->n_rows;
    size_t n_values_t = (size_t)points->n_elem;
    const dtype *p_points = points->memptr();

    // Find longest axis
    dtype x = aabb.at(0, 1) - aabb.at(0, 0);
    dtype y = aabb.at(0, 3) - aabb.at(0, 2);
    dtype z = aabb.at(0, 5) - aabb.at(0, 4);

    if (axis == 0)
    {
        if (z >= y && z >= x)
            axis = 3;
        else if (y >= x && y >= z)
            axis = 2;
        else
            axis = 1;
    }
    else if (axis != 1 && axis != 2 && axis != 3)
        throw std::invalid_argument("Input 'axis' must have values 0, 1, 2 or 3.");

    // Define bounding box A
    dtype x_max = (axis == 1) ? aabb.at(0, 0) + (dtype)0.5 * x : aabb.at(0, 1),
          y_max = (axis == 2) ? aabb.at(0, 2) + (dtype)0.5 * y : aabb.at(0, 3),
          z_max = (axis == 3) ? aabb.at(0, 4) + (dtype)0.5 * z : aabb.at(0, 5);

    // Determine all points that are outside box A
    bool *isB = new bool[n_points_t](); // Init to false

    for (size_t i_val = 0; i_val < n_values_t; ++i_val)
    {
        dtype v = p_points[i_val];         // Mesh value
        size_t i_col = i_val / n_points_t; // Column index in mesh
        size_t i_row = i_val % n_points_t; // Row index in mesh

        if (i_col == 0)
            isB[i_row] = (v > x_max) ? true : isB[i_row];
        else if (i_col == 1)
            isB[i_row] = (v > y_max) ? true : isB[i_row];
        else
            isB[i_row] = (v > z_max) ? true : isB[i_row];
    }

    // Count items in both sub-meshes
    size_t n_pointsA = 0, n_pointsB = 0;
    for (size_t i = 0; i < n_points_t; ++i)
        if (isB[i])
            ++n_pointsB;
        else
            ++n_pointsA;

    // Check if the mesh was split
    if (n_pointsA == 0 || n_pointsB == 0)
        return -axis;

    // Adjust output size
    pointsA->set_size(n_pointsA, 3);
    pointsB->set_size(n_pointsB, 3);

    dtype *p_pointsA = pointsA->memptr();
    dtype *p_pointsB = pointsB->memptr();

    bool write_split_ind = false;
    int *p_split_ind = nullptr;
    if (split_ind != nullptr)
    {
        write_split_ind = true;
        if (split_ind->n_elem != points->n_rows)
            split_ind->zeros(points->n_rows);
        else
            split_ind->zeros();
        p_split_ind = split_ind->memptr();
    }

    // Copy data
    size_t i_pointA = 0, i_pointB = 0;
    for (size_t i_val = 0; i_val < n_values_t; ++i_val)
    {
        dtype v = p_points[i_val];         // Mesh value
        size_t i_row = i_val % n_points_t; // Row index in mesh

        if (isB[i_row])
        {
            p_pointsB[i_pointB++] = v;
            if (write_split_ind)
                p_split_ind[i_row] = 2;
        }
        else
        {
            p_pointsA[i_pointA++] = v;
            if (write_split_ind)
                p_split_ind[i_row] = 1;
        }
    }

    delete[] isB;
    return axis;
}

template int quadriga_lib::point_cloud_split(const arma::Mat<float> *points, arma::Mat<float> *pointsA, arma::Mat<float> *pointsB, int axis, arma::Col<int> *split_ind);
template int quadriga_lib::point_cloud_split(const arma::Mat<double> *points, arma::Mat<double> *pointsA, arma::Mat<double> *pointsB, int axis, arma::Col<int> *split_ind);

// Subdivide rays
template <typename dtype>
size_t quadriga_lib::subdivide_rays(const arma::Mat<dtype> *orig, const arma::Mat<dtype> *trivec, const arma::Mat<dtype> *tridir, const arma::Mat<dtype> *dest,
                                    arma::Mat<dtype> *origN, arma::Mat<dtype> *trivecN, arma::Mat<dtype> *tridirN, arma::Mat<dtype> *destN,
                                    const arma::Col<unsigned> *index, const double ray_offset)
{
    // Check for NULL pointers
    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (trivec == nullptr)
        throw std::invalid_argument("Input 'trivec' cannot be NULL.");
    if (tridir == nullptr)
        throw std::invalid_argument("Input 'tridir' cannot be NULL.");

    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");

    const size_t n_ray = orig->n_rows; // Number of rays
    const size_t n_ray_t = (size_t)n_ray;
    const unsigned n_ray_u = (unsigned)n_ray;

    if (trivec->n_cols != 9)
        throw std::invalid_argument("Input 'trivec' must have 9 columns.");
    if (tridir->n_cols != 6 && tridir->n_cols != 9)
        throw std::invalid_argument("Input 'tridir' must have 6 or 9 columns.");
    if (trivec->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'trivec' dont match.");
    if (tridir->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'tridir' dont match.");

    if (dest != nullptr && !dest->is_empty() && dest->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'dest' does not match number of rows in 'orig'.");
    if (dest != nullptr && !dest->is_empty() && dest->n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");

    size_t n_ind_t = 0;
    unsigned *p_ind;
    if (index != nullptr && index->n_elem != 0)
    {
        n_ind_t = (size_t)index->n_elem;
        const unsigned *tmp = index->memptr();

        for (size_t i = 0; i < n_ind_t; ++i)
            if (tmp[i] >= (unsigned)n_ray_t)
                throw std::invalid_argument("Indices cannot exceed number of rays.");

        p_ind = new unsigned[n_ind_t];
        std::memcpy(p_ind, tmp, n_ind_t * sizeof(unsigned));
    }
    else
    {
        n_ind_t = n_ray_t;
        p_ind = new unsigned[n_ray_t];
        for (unsigned i_ray = 0; i_ray < n_ray_u; ++i_ray)
            p_ind[i_ray] = i_ray;
    }

    // Number of rays in the output
    size_t n_rayN_t = 4 * n_ind_t;
    size_t n_rayN = (size_t)n_rayN_t;

    // Indicator for Cartesian Format
    bool cartesian_format = false;
    if (tridir->n_cols == 9)
        cartesian_format = true;

    // Memory pointers (inputs)
    const dtype *p_orig = orig->memptr();
    const dtype *p_trivec = trivec->memptr();
    const dtype *p_tridir = tridir->memptr();
    const dtype *p_dest = (dest == nullptr || dest->is_empty()) ? nullptr : dest->memptr();

    // Allocate output memory, if needed
    if (origN != nullptr && (origN->n_rows != n_rayN || origN->n_cols != 3))
        origN->set_size(n_rayN, 3);

    if (trivecN != nullptr && (trivecN->n_rows != n_rayN || trivecN->n_cols != 9))
        trivecN->set_size(n_rayN, 9);

    if (tridirN != nullptr && cartesian_format && (tridirN->n_rows != n_rayN || tridirN->n_cols != 9))
        tridirN->set_size(n_rayN, 9);
    else if (tridirN != nullptr && (tridirN->n_rows != n_rayN || tridirN->n_cols != 6))
        tridirN->set_size(n_rayN, 6);

    if (dest != nullptr && destN != nullptr && (destN->n_rows != n_rayN || destN->n_cols != 3))
        destN->set_size(n_rayN, 3);
    else if (destN != nullptr)
        destN->reset();

    // Get output pointers
    dtype *p_origN = (origN == nullptr) ? nullptr : origN->memptr();
    dtype *p_trivecN = (trivecN == nullptr) ? nullptr : trivecN->memptr();
    dtype *p_tridirN = (tridirN == nullptr) ? nullptr : tridirN->memptr();
    dtype *p_destN = (destN == nullptr || destN->is_empty()) ? nullptr : destN->memptr();

    // Iterate through all indices
    for (size_t i_ind = 0; i_ind < n_ind_t; ++i_ind)
    {
        size_t i_ray = p_ind[i_ind];

        // Load beam origin
        double Ox = (double)p_orig[i_ray],
               Oy = (double)p_orig[i_ray + n_ray_t],
               Oz = (double)p_orig[i_ray + 2 * n_ray_t];

        // Load destination and calculate the length from orig to dest
        double length = NAN;
        if (p_dest != nullptr)
        {
            double Ux = (double)p_dest[i_ray] - Ox;
            double Uy = (double)p_dest[i_ray + n_ray_t] - Oy;
            double Uz = (double)p_dest[i_ray + 2 * n_ray_t] - Oz;
            length = std::sqrt(Ux * Ux + Uy * Uy + Uz * Uz);
        }

        // Load the 3 beam vertices
        double W1x = Ox + (double)p_trivec[i_ray],
               W1y = Oy + (double)p_trivec[i_ray + n_ray_t],
               W1z = Oz + (double)p_trivec[i_ray + 2 * n_ray_t];

        double W2x = Ox + (double)p_trivec[i_ray + 3 * n_ray_t],
               W2y = Oy + (double)p_trivec[i_ray + 4 * n_ray_t],
               W2z = Oz + (double)p_trivec[i_ray + 5 * n_ray_t];

        double W3x = Ox + (double)p_trivec[i_ray + 6 * n_ray_t],
               W3y = Oy + (double)p_trivec[i_ray + 7 * n_ray_t],
               W3z = Oz + (double)p_trivec[i_ray + 8 * n_ray_t];

        // Calculate the 3 additional vertices
        double W12x = 0.5 * (W1x + W2x), W12y = 0.5 * (W1y + W2y), W12z = 0.5 * (W1z + W2z);
        double W13x = 0.5 * (W1x + W3x), W13y = 0.5 * (W1y + W3y), W13z = 0.5 * (W1z + W3z);
        double W23x = 0.5 * (W2x + W3x), W23y = 0.5 * (W2y + W3y), W23z = 0.5 * (W2z + W3z);

        // Calculate the direction vectors at the vertices
        double D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z, scl;
        if (cartesian_format)
        {
            D1x = (double)p_tridir[i_ray];
            D1y = (double)p_tridir[i_ray + n_ray_t];
            D1z = (double)p_tridir[i_ray + 2 * n_ray_t];

            scl = D1x * D1x + D1y * D1y + D1z * D1z;
            if (std::abs(scl - 1.0) > 2.0e-7) // Normalize
                scl = 1.0 / std::sqrt(scl), D1x *= scl, D1y *= scl, D1z *= scl;

            D2x = (double)p_tridir[i_ray + 3 * n_ray_t];
            D2y = (double)p_tridir[i_ray + 4 * n_ray_t];
            D2z = (double)p_tridir[i_ray + 5 * n_ray_t];

            scl = D2x * D2x + D2y * D2y + D2z * D2z;
            if (std::abs(scl - 1.0) > 2.0e-7) // Normalize
                scl = 1.0 / std::sqrt(scl), D2x *= scl, D2y *= scl, D2z *= scl;

            D3x = (double)p_tridir[i_ray + 6 * n_ray_t];
            D3y = (double)p_tridir[i_ray + 7 * n_ray_t];
            D3z = (double)p_tridir[i_ray + 8 * n_ray_t];

            scl = D3x * D3x + D3y * D3y + D3z * D3z;
            if (std::abs(scl - 1.0) > 2.0e-7) // Normalize
                scl = 1.0 / std::sqrt(scl), D3x *= scl, D3y *= scl, D3z *= scl;
        }
        else // Spherical format
        {
            double az = (double)p_tridir[i_ray],
                   el = (double)p_tridir[i_ray + n_ray_t];

            scl = std::cos(el);
            D1x = std::cos(az) * scl, D1y = std::sin(az) * scl, D1z = std::sin(el);

            az = (double)p_tridir[i_ray + 2 * n_ray_t];
            el = (double)p_tridir[i_ray + 3 * n_ray_t];

            scl = std::cos(el);
            D2x = std::cos(az) * scl, D2y = std::sin(az) * scl, D2z = std::sin(el);

            az = (double)p_tridir[i_ray + 4 * n_ray_t];
            el = (double)p_tridir[i_ray + 5 * n_ray_t];

            scl = std::cos(el);
            D3x = std::cos(az) * scl, D3y = std::sin(az) * scl, D3z = std::sin(el);
        }

        // Calculate the directions at the 3 additional vertices
        double D12x = 0.5 * (D1x + D2x), D12y = 0.5 * (D1y + D2y), D12z = 0.5 * (D1z + D2z);
        scl = 1.0 / std::sqrt(D12x * D12x + D12y * D12y + D12z * D12z), D12x *= scl, D12y *= scl, D12z *= scl;

        double D13x = 0.5 * (D1x + D3x), D13y = 0.5 * (D1y + D3y), D13z = 0.5 * (D1z + D3z);
        scl = 1.0 / std::sqrt(D13x * D13x + D13y * D13y + D13z * D13z), D13x *= scl, D13y *= scl, D13z *= scl;

        double D23x = 0.5 * (D2x + D3x), D23y = 0.5 * (D2y + D3y), D23z = 0.5 * (D2z + D3z);
        scl = 1.0 / std::sqrt(D23x * D23x + D23y * D23y + D23z * D23z), D23x *= scl, D23y *= scl, D23z *= scl;

        // Convert to Spheric coordinates
        dtype az12 = NAN, el12 = NAN, az13 = NAN, el13 = NAN, az23 = NAN, el23 = NAN;
        if (p_tridirN != nullptr && !cartesian_format)
        {
            az12 = (dtype)std::atan2(D12y, D12x), el12 = (dtype)std::asin(D12z);
            az13 = (dtype)std::atan2(D13y, D13x), el13 = (dtype)std::asin(D13z);
            az23 = (dtype)std::atan2(D23y, D23x), el23 = (dtype)std::asin(D23z);
        }

        // Create the 4 sub-triangles
        for (size_t i_sub = 0; i_sub < 4; ++i_sub)
        {
            // Index of the ray in the output
            size_t i_rayN = 4 * i_ind + i_sub;

            double w1x, w1y, w1z, w2x, w2y, w2z, w3x, w3y, w3z;
            double d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z;

            if (i_sub == 0)
            {
                w1x = W1x, w1y = W1y, w1z = W1z, w2x = W12x, w2y = W12y, w2z = W12z, w3x = W13x, w3y = W13y, w3z = W13z;
                d1x = D1x, d1y = D1y, d1z = D1z, d2x = D12x, d2y = D12y, d2z = D12z, d3x = D13x, d3y = D13y, d3z = D13z;

                if (p_tridirN != nullptr && !cartesian_format)
                {
                    p_tridirN[i_rayN] = p_tridir[i_ray];
                    p_tridirN[i_rayN + n_rayN_t] = p_tridir[i_ray + n_ray_t];
                    p_tridirN[i_rayN + 2 * n_rayN_t] = az12;
                    p_tridirN[i_rayN + 3 * n_rayN_t] = el12;
                    p_tridirN[i_rayN + 4 * n_rayN_t] = az13;
                    p_tridirN[i_rayN + 5 * n_rayN_t] = el13;
                }
            }
            else if (i_sub == 1)
            {
                w1x = W13x, w1y = W13y, w1z = W13z, w2x = W12x, w2y = W12y, w2z = W12z, w3x = W23x, w3y = W23y, w3z = W23z;
                d1x = D13x, d1y = D13y, d1z = D13z, d2x = D12x, d2y = D12y, d2z = D12z, d3x = D23x, d3y = D23y, d3z = D23z;

                if (p_tridirN != nullptr && !cartesian_format)
                {
                    p_tridirN[i_rayN] = az13;
                    p_tridirN[i_rayN + n_rayN_t] = el13;
                    p_tridirN[i_rayN + 2 * n_rayN_t] = az12;
                    p_tridirN[i_rayN + 3 * n_rayN_t] = el12;
                    p_tridirN[i_rayN + 4 * n_rayN_t] = az23;
                    p_tridirN[i_rayN + 5 * n_rayN_t] = el23;
                }
            }
            else if (i_sub == 2)
            {
                w1x = W13x, w1y = W13y, w1z = W13z, w2x = W23x, w2y = W23y, w2z = W23z, w3x = W3x, w3y = W3y, w3z = W3z;
                d1x = D13x, d1y = D13y, d1z = D13z, d2x = D23x, d2y = D23y, d2z = D23z, d3x = D3x, d3y = D3y, d3z = D3z;

                if (p_tridirN != nullptr && !cartesian_format)
                {
                    p_tridirN[i_rayN] = az13;
                    p_tridirN[i_rayN + n_rayN_t] = el13;
                    p_tridirN[i_rayN + 2 * n_rayN_t] = az23;
                    p_tridirN[i_rayN + 3 * n_rayN_t] = el23;
                    p_tridirN[i_rayN + 4 * n_rayN_t] = p_tridir[i_ray + 4 * n_ray_t];
                    p_tridirN[i_rayN + 5 * n_rayN_t] = p_tridir[i_ray + 5 * n_ray_t];
                }
            }
            else if (i_sub == 3)
            {
                w1x = W12x, w1y = W12y, w1z = W12z, w2x = W2x, w2y = W2y, w2z = W2z, w3x = W23x, w3y = W23y, w3z = W23z;
                d1x = D12x, d1y = D12y, d1z = D12z, d2x = D2x, d2y = D2y, d2z = D2z, d3x = D23x, d3y = D23y, d3z = D23z;

                if (p_tridirN != nullptr && !cartesian_format)
                {
                    p_tridirN[i_rayN] = az12;
                    p_tridirN[i_rayN + n_rayN_t] = el12;
                    p_tridirN[i_rayN + 2 * n_rayN_t] = p_tridir[i_ray + 2 * n_ray_t];
                    p_tridirN[i_rayN + 3 * n_rayN_t] = p_tridir[i_ray + 3 * n_ray_t];
                    p_tridirN[i_rayN + 4 * n_rayN_t] = az23;
                    p_tridirN[i_rayN + 5 * n_rayN_t] = el23;
                }
            }

            // Write "tridir" for Cartesian format
            if (p_tridirN != nullptr && cartesian_format)
            {
                p_tridirN[i_rayN] = (dtype)d1x;
                p_tridirN[i_rayN + n_rayN_t] = (dtype)d1y;
                p_tridirN[i_rayN + 2 * n_rayN_t] = (dtype)d1z;
                p_tridirN[i_rayN + 3 * n_rayN_t] = (dtype)d2x;
                p_tridirN[i_rayN + 4 * n_rayN_t] = (dtype)d2y;
                p_tridirN[i_rayN + 5 * n_rayN_t] = (dtype)d2z;
                p_tridirN[i_rayN + 6 * n_rayN_t] = (dtype)d3x;
                p_tridirN[i_rayN + 7 * n_rayN_t] = (dtype)d3y;
                p_tridirN[i_rayN + 8 * n_rayN_t] = (dtype)d3z;
            }

            // Calculate center point
            double ox = 0.333333333333333 * (w1x + w2x + w3x),
                   oy = 0.333333333333333 * (w1y + w2y + w3y),
                   oz = 0.333333333333333 * (w1z + w2z + w3z);

            // Calculate ray direction at center point
            double dx = 0.333333333333333 * (d1x + d2x + d3x),
                   dy = 0.333333333333333 * (d1y + d2y + d3y),
                   dz = 0.333333333333333 * (d1z + d2z + d3z);

            double scl = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
            dx *= scl, dy *= scl, dz *= scl;

            // Write new destination
            if (p_destN != nullptr)
            {
                double tx = ox + length * dx,
                       ty = oy + length * dy,
                       tz = oz + length * dz;

                p_destN[i_rayN] = (dtype)tx;
                p_destN[i_rayN + n_rayN_t] = (dtype)ty;
                p_destN[i_rayN + 2 * n_rayN_t] = (dtype)tz;
            }

            // Calculate new origin point (consider ray offset)
            ox += ray_offset * dx, oy += ray_offset * dy, oz += ray_offset * dz;

            // Write new origin point
            if (p_origN != nullptr)
            {
                p_origN[i_rayN] = (dtype)ox;
                p_origN[i_rayN + n_rayN_t] = (dtype)oy;
                p_origN[i_rayN + 2 * n_rayN_t] = (dtype)oz;
            }

            // Write new trivec
            if (p_trivecN != nullptr)
            {
                double tx = w1x - ox, ty = w1y - oy, tz = w1z - oz;

                p_trivecN[i_rayN] = (dtype)tx;
                p_trivecN[i_rayN + n_rayN_t] = (dtype)ty;
                p_trivecN[i_rayN + 2 * n_rayN_t] = (dtype)tz;

                tx = w2x - ox, ty = w2y - oy, tz = w2z - oz;

                p_trivecN[i_rayN + 3 * n_rayN_t] = (dtype)tx;
                p_trivecN[i_rayN + 4 * n_rayN_t] = (dtype)ty;
                p_trivecN[i_rayN + 5 * n_rayN_t] = (dtype)tz;

                tx = w3x - ox, ty = w3y - oy, tz = w3z - oz;

                p_trivecN[i_rayN + 6 * n_rayN_t] = (dtype)tx;
                p_trivecN[i_rayN + 7 * n_rayN_t] = (dtype)ty;
                p_trivecN[i_rayN + 8 * n_rayN_t] = (dtype)tz;
            }
        }
    }

    delete[] p_ind;
    return n_rayN_t;
}

template size_t quadriga_lib::subdivide_rays(const arma::Mat<float> *orig, const arma::Mat<float> *trivec, const arma::Mat<float> *tridir, const arma::Mat<float> *dest,
                                             arma::Mat<float> *origN, arma::Mat<float> *trivecN, arma::Mat<float> *tridirN, arma::Mat<float> *destN,
                                             const arma::Col<unsigned> *index, const double ray_offset);

template size_t quadriga_lib::subdivide_rays(const arma::Mat<double> *orig, const arma::Mat<double> *trivec, const arma::Mat<double> *tridir, const arma::Mat<double> *dest,
                                             arma::Mat<double> *origN, arma::Mat<double> *trivecN, arma::Mat<double> *tridirN, arma::Mat<double> *destN,
                                             const arma::Col<unsigned> *index, const double ray_offset);

// Subdivide triangles into smaller triangles
template <typename dtype>
size_t quadriga_lib::subdivide_triangles(arma::uword n_div, const arma::Mat<dtype> *triangles_in, arma::Mat<dtype> *triangles_out,
                                         const arma::Mat<dtype> *mtl_prop, arma::Mat<dtype> *mtl_prop_out)
{
    if (n_div == 0)
        throw std::invalid_argument("Input 'n_div' cannot be 0.");

    if (triangles_in == nullptr || triangles_in->n_elem == 0)
        throw std::invalid_argument("Input 'triangles_in' cannot be NULL.");

    if (triangles_in->n_cols != 9)
        throw std::invalid_argument("Input 'triangles_in' must have 9 columns.");

    if (triangles_out == nullptr)
        throw std::invalid_argument("Output 'triangles_out' cannot be NULL.");

    size_t n_div_t = (size_t)n_div;
    size_t n_triangles_in = (size_t)triangles_in->n_rows;
    size_t n_triangles_out = n_triangles_in * n_div_t * n_div_t;

    if (triangles_out->n_cols != 9 || triangles_out->n_rows != n_triangles_out)
        triangles_out->set_size(n_triangles_out, 9);

    bool process_mtl_prop = (mtl_prop != nullptr) && (mtl_prop_out != nullptr) && (mtl_prop->n_elem != 0);

    if (process_mtl_prop)
    {
        if (mtl_prop->n_cols != 5)
            throw std::invalid_argument("Input 'mtl_prop' must have 5 columns.");

        if (mtl_prop->n_rows != n_triangles_in)
            throw std::invalid_argument("Number of rows in 'triangles_in' and 'mtl_prop' dont match.");

        if (mtl_prop_out->n_cols != 5 || mtl_prop_out->n_rows != n_triangles_out)
            mtl_prop_out->set_size(n_triangles_out, 5);
    }

    // Process each triangle
    size_t cnt = 0;                                       // Counter
    dtype stp = dtype(1.0) / dtype(n_div);                // Step size
    const dtype *p_triangles_in = triangles_in->memptr(); // Pointer to input memory
    dtype *p_triangles_out = triangles_out->memptr();     // Pointer to output memory
    const dtype *p_mtl_prop = process_mtl_prop ? mtl_prop->memptr() : nullptr;
    dtype *p_mtl_prop_out = process_mtl_prop ? mtl_prop_out->memptr() : nullptr;

    for (size_t n = 0; n < n_triangles_in; ++n)
    {
        // Read current triangle vertices
        dtype v1x = p_triangles_in[n];
        dtype v1y = p_triangles_in[n + n_triangles_in];
        dtype v1z = p_triangles_in[n + 2 * n_triangles_in];
        dtype e12x = p_triangles_in[n + 3 * n_triangles_in] - v1x;
        dtype e12y = p_triangles_in[n + 4 * n_triangles_in] - v1y;
        dtype e12z = p_triangles_in[n + 5 * n_triangles_in] - v1z;
        dtype e13x = p_triangles_in[n + 6 * n_triangles_in] - v1x;
        dtype e13y = p_triangles_in[n + 7 * n_triangles_in] - v1y;
        dtype e13z = p_triangles_in[n + 8 * n_triangles_in] - v1z;

        // Read current material
        dtype mtl[5];
        if (process_mtl_prop)
            mtl[0] = p_mtl_prop[n],
            mtl[1] = p_mtl_prop[n + n_triangles_in],
            mtl[2] = p_mtl_prop[n + 2 * n_triangles_in],
            mtl[3] = p_mtl_prop[n + 3 * n_triangles_in],
            mtl[4] = p_mtl_prop[n + 4 * n_triangles_in];

        for (size_t u = 0; u < n_div_t; ++u)
        {
            dtype ful = dtype(u) * stp;
            dtype fuu = ful + stp;

            for (size_t v = 0; v < n_div_t - u; ++v)
            {
                dtype fvl = dtype(v) * stp;
                dtype fvu = fvl + stp;

                // Lower triangle first vertex
                p_triangles_out[cnt] = v1x + fvl * e12x + ful * e13x;                       // w1x
                p_triangles_out[cnt + n_triangles_out] = v1y + fvl * e12y + ful * e13y;     // w1y
                p_triangles_out[cnt + 2 * n_triangles_out] = v1z + fvl * e12z + ful * e13z; // w1z

                // Lower triangle second vertex
                p_triangles_out[cnt + 3 * n_triangles_out] = v1x + fvu * e12x + ful * e13x; // w2x
                p_triangles_out[cnt + 4 * n_triangles_out] = v1y + fvu * e12y + ful * e13y; // w2y
                p_triangles_out[cnt + 5 * n_triangles_out] = v1z + fvu * e12z + ful * e13z; // w2z

                // Lower triangle third vertex
                p_triangles_out[cnt + 6 * n_triangles_out] = v1x + fvl * e12x + fuu * e13x; // w2x
                p_triangles_out[cnt + 7 * n_triangles_out] = v1y + fvl * e12y + fuu * e13y; // w2y
                p_triangles_out[cnt + 8 * n_triangles_out] = v1z + fvl * e12z + fuu * e13z; // w2z

                // Material
                if (process_mtl_prop)
                    p_mtl_prop_out[cnt] = mtl[0],
                    p_mtl_prop_out[cnt + n_triangles_out] = mtl[1],
                    p_mtl_prop_out[cnt + 2 * n_triangles_out] = mtl[2],
                    p_mtl_prop_out[cnt + 3 * n_triangles_out] = mtl[3],
                    p_mtl_prop_out[cnt + 4 * n_triangles_out] = mtl[4];

                ++cnt;

                if (v < n_div - u - 1)
                {
                    // Upper triangle first vertex
                    p_triangles_out[cnt] = v1x + fvl * e12x + fuu * e13x;                       // w1x
                    p_triangles_out[cnt + n_triangles_out] = v1y + fvl * e12y + fuu * e13y;     // w1y
                    p_triangles_out[cnt + 2 * n_triangles_out] = v1z + fvl * e12z + fuu * e13z; // w1z

                    // Upper triangle second vertex
                    p_triangles_out[cnt + 6 * n_triangles_out] = v1x + fvu * e12x + fuu * e13x; // w2x
                    p_triangles_out[cnt + 7 * n_triangles_out] = v1y + fvu * e12y + fuu * e13y; // w2y
                    p_triangles_out[cnt + 8 * n_triangles_out] = v1z + fvu * e12z + fuu * e13z; // w2z

                    // Upper triangle third vertex
                    p_triangles_out[cnt + 3 * n_triangles_out] = v1x + fvu * e12x + ful * e13x; // w2x
                    p_triangles_out[cnt + 4 * n_triangles_out] = v1y + fvu * e12y + ful * e13y; // w2y
                    p_triangles_out[cnt + 5 * n_triangles_out] = v1z + fvu * e12z + ful * e13z; // w2z

                    // Material
                    if (process_mtl_prop)
                        p_mtl_prop_out[cnt] = mtl[0],
                        p_mtl_prop_out[cnt + n_triangles_out] = mtl[1],
                        p_mtl_prop_out[cnt + 2 * n_triangles_out] = mtl[2],
                        p_mtl_prop_out[cnt + 3 * n_triangles_out] = mtl[3],
                        p_mtl_prop_out[cnt + 4 * n_triangles_out] = mtl[4];

                    ++cnt;
                }
            }
        }
    }
    return n_triangles_out;
}

template size_t quadriga_lib::subdivide_triangles(arma::uword n_div, const arma::Mat<float> *triangles_in, arma::Mat<float> *triangles_out,
                                                  const arma::Mat<float> *mtl_prop, arma::Mat<float> *mtl_prop_out);

template size_t quadriga_lib::subdivide_triangles(arma::uword n_div, const arma::Mat<double> *triangles_in, arma::Mat<double> *triangles_out,
                                                  const arma::Mat<double> *mtl_prop, arma::Mat<double> *mtl_prop_out);

// Generate colormap
arma::uchar_mat quadriga_lib::colormap(std::string map)
{
    /*! MATLAB code to generate colormap:
        clc
        cmap_values = round( feval('jet', 64)*255 );
        str = '{';
        for i = 1:size(cmap_values,1)
            str = [str, sprintf('{%d,%d,%d},',cmap_values(i,:))];
            if (mod(i,8) == 0)
                str = [str, '\n'];
            end
        end
        str = [str(1:end-3), '};\n'];
        fprintf(str)
    !*/

    arma::uchar_mat cmap(64, 3);
    if (map == "jet")
        cmap = {{0, 0, 143}, {0, 0, 159}, {0, 0, 175}, {0, 0, 191}, {0, 0, 207}, {0, 0, 223}, {0, 0, 239}, {0, 0, 255}, {0, 16, 255}, {0, 32, 255}, {0, 48, 255}, {0, 64, 255}, {0, 80, 255}, {0, 96, 255}, {0, 112, 255}, {0, 128, 255}, {0, 143, 255}, {0, 159, 255}, {0, 175, 255}, {0, 191, 255}, {0, 207, 255}, {0, 223, 255}, {0, 239, 255}, {0, 255, 255}, {16, 255, 239}, {32, 255, 223}, {48, 255, 207}, {64, 255, 191}, {80, 255, 175}, {96, 255, 159}, {112, 255, 143}, {128, 255, 128}, {143, 255, 112}, {159, 255, 96}, {175, 255, 80}, {191, 255, 64}, {207, 255, 48}, {223, 255, 32}, {239, 255, 16}, {255, 255, 0}, {255, 239, 0}, {255, 223, 0}, {255, 207, 0}, {255, 191, 0}, {255, 175, 0}, {255, 159, 0}, {255, 143, 0}, {255, 128, 0}, {255, 112, 0}, {255, 96, 0}, {255, 80, 0}, {255, 64, 0}, {255, 48, 0}, {255, 32, 0}, {255, 16, 0}, {255, 0, 0}, {239, 0, 0}, {223, 0, 0}, {207, 0, 0}, {191, 0, 0}, {175, 0, 0}, {159, 0, 0}, {143, 0, 0}, {128, 0, 0}};
    else if (map == "parula")
        cmap = {{62, 38, 168}, {64, 42, 180}, {66, 46, 192}, {68, 50, 203}, {69, 55, 213}, {70, 60, 222}, {71, 65, 229}, {71, 71, 235}, {72, 77, 240}, {72, 82, 244}, {71, 88, 248}, {70, 94, 251}, {69, 99, 253}, {66, 105, 254}, {62, 111, 255}, {56, 117, 254}, {50, 124, 252}, {47, 129, 250}, {46, 135, 247}, {45, 140, 243}, {43, 145, 239}, {39, 151, 235}, {37, 155, 232}, {35, 160, 229}, {32, 165, 227}, {28, 169, 223}, {24, 173, 219}, {18, 177, 214}, {8, 181, 208}, {1, 184, 202}, {2, 186, 195}, {11, 189, 189}, {25, 191, 182}, {36, 193, 174}, {44, 196, 167}, {49, 198, 159}, {55, 200, 151}, {63, 202, 142}, {74, 203, 132}, {87, 204, 122}, {100, 205, 111}, {114, 205, 100}, {129, 204, 89}, {143, 203, 78}, {157, 201, 67}, {171, 199, 57}, {185, 196, 49}, {197, 194, 42}, {209, 191, 39}, {220, 189, 41}, {230, 187, 45}, {240, 186, 54}, {248, 186, 61}, {254, 190, 60}, {254, 195, 56}, {254, 201, 52}, {252, 207, 48}, {250, 214, 45}, {247, 220, 42}, {245, 227, 39}, {245, 233, 36}, {246, 239, 32}, {247, 245, 27}, {249, 251, 21}};
    else if (map == "winter")
        cmap = {{0, 0, 255}, {0, 4, 253}, {0, 8, 251}, {0, 12, 249}, {0, 16, 247}, {0, 20, 245}, {0, 24, 243}, {0, 28, 241}, {0, 32, 239}, {0, 36, 237}, {0, 40, 235}, {0, 45, 233}, {0, 49, 231}, {0, 53, 229}, {0, 57, 227}, {0, 61, 225}, {0, 65, 223}, {0, 69, 221}, {0, 73, 219}, {0, 77, 217}, {0, 81, 215}, {0, 85, 213}, {0, 89, 210}, {0, 93, 208}, {0, 97, 206}, {0, 101, 204}, {0, 105, 202}, {0, 109, 200}, {0, 113, 198}, {0, 117, 196}, {0, 121, 194}, {0, 125, 192}, {0, 130, 190}, {0, 134, 188}, {0, 138, 186}, {0, 142, 184}, {0, 146, 182}, {0, 150, 180}, {0, 154, 178}, {0, 158, 176}, {0, 162, 174}, {0, 166, 172}, {0, 170, 170}, {0, 174, 168}, {0, 178, 166}, {0, 182, 164}, {0, 186, 162}, {0, 190, 160}, {0, 194, 158}, {0, 198, 156}, {0, 202, 154}, {0, 206, 152}, {0, 210, 150}, {0, 215, 148}, {0, 219, 146}, {0, 223, 144}, {0, 227, 142}, {0, 231, 140}, {0, 235, 138}, {0, 239, 136}, {0, 243, 134}, {0, 247, 132}, {0, 251, 130}, {0, 255, 128}};
    else if (map == "hot")
        cmap = {{11, 0, 0}, {21, 0, 0}, {32, 0, 0}, {43, 0, 0}, {53, 0, 0}, {64, 0, 0}, {74, 0, 0}, {85, 0, 0}, {96, 0, 0}, {106, 0, 0}, {117, 0, 0}, {128, 0, 0}, {138, 0, 0}, {149, 0, 0}, {159, 0, 0}, {170, 0, 0}, {181, 0, 0}, {191, 0, 0}, {202, 0, 0}, {213, 0, 0}, {223, 0, 0}, {234, 0, 0}, {244, 0, 0}, {255, 0, 0}, {255, 11, 0}, {255, 21, 0}, {255, 32, 0}, {255, 43, 0}, {255, 53, 0}, {255, 64, 0}, {255, 74, 0}, {255, 85, 0}, {255, 96, 0}, {255, 106, 0}, {255, 117, 0}, {255, 128, 0}, {255, 138, 0}, {255, 149, 0}, {255, 159, 0}, {255, 170, 0}, {255, 181, 0}, {255, 191, 0}, {255, 202, 0}, {255, 213, 0}, {255, 223, 0}, {255, 234, 0}, {255, 244, 0}, {255, 255, 0}, {255, 255, 16}, {255, 255, 32}, {255, 255, 48}, {255, 255, 64}, {255, 255, 80}, {255, 255, 96}, {255, 255, 112}, {255, 255, 128}, {255, 255, 143}, {255, 255, 159}, {255, 255, 175}, {255, 255, 191}, {255, 255, 207}, {255, 255, 223}, {255, 255, 239}, {255, 255, 255}};
    else if (map == "turbo")
        cmap = {{48, 18, 59}, {53, 30, 89}, {57, 42, 116}, {61, 54, 140}, {64, 65, 163}, {67, 76, 183}, {68, 87, 200}, {70, 98, 215}, {70, 108, 228}, {71, 119, 239}, {70, 129, 247}, {69, 139, 253}, {65, 149, 255}, {60, 159, 253}, {53, 170, 249}, {46, 180, 242}, {39, 189, 234}, {33, 199, 224}, {28, 207, 213}, {24, 215, 202}, {24, 222, 192}, {26, 228, 182}, {33, 234, 172}, {42, 239, 161}, {54, 243, 148}, {67, 247, 134}, {83, 250, 121}, {98, 252, 107}, {114, 254, 94}, {130, 255, 82}, {144, 255, 72}, {157, 254, 64}, {168, 252, 58}, {179, 248, 54}, {189, 244, 52}, {200, 239, 52}, {209, 233, 53}, {218, 226, 54}, {227, 219, 56}, {234, 211, 57}, {241, 203, 58}, {246, 195, 58}, {250, 186, 57}, {252, 177, 54}, {254, 166, 50}, {254, 155, 45}, {253, 143, 40}, {252, 131, 35}, {249, 119, 30}, {246, 107, 25}, {242, 95, 20}, {237, 84, 15}, {231, 74, 12}, {225, 66, 9}, {218, 57, 7}, {211, 50, 5}, {202, 42, 4}, {193, 35, 2}, {183, 29, 2}, {173, 23, 1}, {161, 18, 1}, {149, 13, 1}, {136, 8, 2}, {122, 4, 3}};
    else if (map == "copper")
        cmap = {{0, 0, 0}, {5, 3, 2}, {10, 6, 4}, {15, 9, 6}, {20, 13, 8}, {25, 16, 10}, {30, 19, 12}, {35, 22, 14}, {40, 25, 16}, {46, 28, 18}, {51, 32, 20}, {56, 35, 22}, {61, 38, 24}, {66, 41, 26}, {71, 44, 28}, {76, 47, 30}, {81, 51, 32}, {86, 54, 34}, {91, 57, 36}, {96, 60, 38}, {101, 63, 40}, {106, 66, 42}, {111, 70, 44}, {116, 73, 46}, {121, 76, 48}, {126, 79, 50}, {132, 82, 52}, {137, 85, 54}, {142, 89, 56}, {147, 92, 58}, {152, 95, 60}, {157, 98, 62}, {162, 101, 64}, {167, 104, 66}, {172, 108, 68}, {177, 111, 70}, {182, 114, 72}, {187, 117, 75}, {192, 120, 77}, {197, 123, 79}, {202, 126, 81}, {207, 130, 83}, {212, 133, 85}, {218, 136, 87}, {223, 139, 89}, {228, 142, 91}, {233, 145, 93}, {238, 149, 95}, {243, 152, 97}, {248, 155, 99}, {253, 158, 101}, {255, 161, 103}, {255, 164, 105}, {255, 168, 107}, {255, 171, 109}, {255, 174, 111}, {255, 177, 113}, {255, 180, 115}, {255, 183, 117}, {255, 187, 119}, {255, 190, 121}, {255, 193, 123}, {255, 196, 125}, {255, 199, 127}};
    else if (map == "spring")
        cmap = {{255, 0, 255}, {255, 4, 251}, {255, 8, 247}, {255, 12, 243}, {255, 16, 239}, {255, 20, 235}, {255, 24, 231}, {255, 28, 227}, {255, 32, 223}, {255, 36, 219}, {255, 40, 215}, {255, 45, 210}, {255, 49, 206}, {255, 53, 202}, {255, 57, 198}, {255, 61, 194}, {255, 65, 190}, {255, 69, 186}, {255, 73, 182}, {255, 77, 178}, {255, 81, 174}, {255, 85, 170}, {255, 89, 166}, {255, 93, 162}, {255, 97, 158}, {255, 101, 154}, {255, 105, 150}, {255, 109, 146}, {255, 113, 142}, {255, 117, 138}, {255, 121, 134}, {255, 125, 130}, {255, 130, 125}, {255, 134, 121}, {255, 138, 117}, {255, 142, 113}, {255, 146, 109}, {255, 150, 105}, {255, 154, 101}, {255, 158, 97}, {255, 162, 93}, {255, 166, 89}, {255, 170, 85}, {255, 174, 81}, {255, 178, 77}, {255, 182, 73}, {255, 186, 69}, {255, 190, 65}, {255, 194, 61}, {255, 198, 57}, {255, 202, 53}, {255, 206, 49}, {255, 210, 45}, {255, 215, 40}, {255, 219, 36}, {255, 223, 32}, {255, 227, 28}, {255, 231, 24}, {255, 235, 20}, {255, 239, 16}, {255, 243, 12}, {255, 247, 8}, {255, 251, 4}, {255, 255, 0}};
    else if (map == "cool")
        cmap = {{0, 255, 255}, {4, 251, 255}, {8, 247, 255}, {12, 243, 255}, {16, 239, 255}, {20, 235, 255}, {24, 231, 255}, {28, 227, 255}, {32, 223, 255}, {36, 219, 255}, {40, 215, 255}, {45, 210, 255}, {49, 206, 255}, {53, 202, 255}, {57, 198, 255}, {61, 194, 255}, {65, 190, 255}, {69, 186, 255}, {73, 182, 255}, {77, 178, 255}, {81, 174, 255}, {85, 170, 255}, {89, 166, 255}, {93, 162, 255}, {97, 158, 255}, {101, 154, 255}, {105, 150, 255}, {109, 146, 255}, {113, 142, 255}, {117, 138, 255}, {121, 134, 255}, {125, 130, 255}, {130, 125, 255}, {134, 121, 255}, {138, 117, 255}, {142, 113, 255}, {146, 109, 255}, {150, 105, 255}, {154, 101, 255}, {158, 97, 255}, {162, 93, 255}, {166, 89, 255}, {170, 85, 255}, {174, 81, 255}, {178, 77, 255}, {182, 73, 255}, {186, 69, 255}, {190, 65, 255}, {194, 61, 255}, {198, 57, 255}, {202, 53, 255}, {206, 49, 255}, {210, 45, 255}, {215, 40, 255}, {219, 36, 255}, {223, 32, 255}, {227, 28, 255}, {231, 24, 255}, {235, 20, 255}, {239, 16, 255}, {243, 12, 255}, {247, 8, 255}, {251, 4, 255}, {255, 0, 255}};
    else if (map == "gray")
        cmap = {{0, 0, 0}, {4, 4, 4}, {8, 8, 8}, {12, 12, 12}, {16, 16, 16}, {20, 20, 20}, {24, 24, 24}, {28, 28, 28}, {32, 32, 32}, {36, 36, 36}, {40, 40, 40}, {45, 45, 45}, {49, 49, 49}, {53, 53, 53}, {57, 57, 57}, {61, 61, 61}, {65, 65, 65}, {69, 69, 69}, {73, 73, 73}, {77, 77, 77}, {81, 81, 81}, {85, 85, 85}, {89, 89, 89}, {93, 93, 93}, {97, 97, 97}, {101, 101, 101}, {105, 105, 105}, {109, 109, 109}, {113, 113, 113}, {117, 117, 117}, {121, 121, 121}, {125, 125, 125}, {130, 130, 130}, {134, 134, 134}, {138, 138, 138}, {142, 142, 142}, {146, 146, 146}, {150, 150, 150}, {154, 154, 154}, {158, 158, 158}, {162, 162, 162}, {166, 166, 166}, {170, 170, 170}, {174, 174, 174}, {178, 178, 178}, {182, 182, 182}, {186, 186, 186}, {190, 190, 190}, {194, 194, 194}, {198, 198, 198}, {202, 202, 202}, {206, 206, 206}, {210, 210, 210}, {215, 215, 215}, {219, 219, 219}, {223, 223, 223}, {227, 227, 227}, {231, 231, 231}, {235, 235, 235}, {239, 239, 239}, {243, 243, 243}, {247, 247, 247}, {251, 251, 251}, {255, 255, 255}};
    else if (map == "autumn")
        cmap = {{255, 0, 0}, {255, 4, 0}, {255, 8, 0}, {255, 12, 0}, {255, 16, 0}, {255, 20, 0}, {255, 24, 0}, {255, 28, 0}, {255, 32, 0}, {255, 36, 0}, {255, 40, 0}, {255, 45, 0}, {255, 49, 0}, {255, 53, 0}, {255, 57, 0}, {255, 61, 0}, {255, 65, 0}, {255, 69, 0}, {255, 73, 0}, {255, 77, 0}, {255, 81, 0}, {255, 85, 0}, {255, 89, 0}, {255, 93, 0}, {255, 97, 0}, {255, 101, 0}, {255, 105, 0}, {255, 109, 0}, {255, 113, 0}, {255, 117, 0}, {255, 121, 0}, {255, 125, 0}, {255, 130, 0}, {255, 134, 0}, {255, 138, 0}, {255, 142, 0}, {255, 146, 0}, {255, 150, 0}, {255, 154, 0}, {255, 158, 0}, {255, 162, 0}, {255, 166, 0}, {255, 170, 0}, {255, 174, 0}, {255, 178, 0}, {255, 182, 0}, {255, 186, 0}, {255, 190, 0}, {255, 194, 0}, {255, 198, 0}, {255, 202, 0}, {255, 206, 0}, {255, 210, 0}, {255, 215, 0}, {255, 219, 0}, {255, 223, 0}, {255, 227, 0}, {255, 231, 0}, {255, 235, 0}, {255, 239, 0}, {255, 243, 0}, {255, 247, 0}, {255, 251, 0}, {255, 255, 0}};
    else if (map == "summer")
        cmap = {{0, 128, 102}, {4, 130, 102}, {8, 132, 102}, {12, 134, 102}, {16, 136, 102}, {20, 138, 102}, {24, 140, 102}, {28, 142, 102}, {32, 144, 102}, {36, 146, 102}, {40, 148, 102}, {45, 150, 102}, {49, 152, 102}, {53, 154, 102}, {57, 156, 102}, {61, 158, 102}, {65, 160, 102}, {69, 162, 102}, {73, 164, 102}, {77, 166, 102}, {81, 168, 102}, {85, 170, 102}, {89, 172, 102}, {93, 174, 102}, {97, 176, 102}, {101, 178, 102}, {105, 180, 102}, {109, 182, 102}, {113, 184, 102}, {117, 186, 102}, {121, 188, 102}, {125, 190, 102}, {130, 192, 102}, {134, 194, 102}, {138, 196, 102}, {142, 198, 102}, {146, 200, 102}, {150, 202, 102}, {154, 204, 102}, {158, 206, 102}, {162, 208, 102}, {166, 210, 102}, {170, 212, 102}, {174, 215, 102}, {178, 217, 102}, {182, 219, 102}, {186, 221, 102}, {190, 223, 102}, {194, 225, 102}, {198, 227, 102}, {202, 229, 102}, {206, 231, 102}, {210, 233, 102}, {215, 235, 102}, {219, 237, 102}, {223, 239, 102}, {227, 241, 102}, {231, 243, 102}, {235, 245, 102}, {239, 247, 102}, {243, 249, 102}, {247, 251, 102}, {251, 253, 102}, {255, 255, 102}};
    else
        throw std::invalid_argument("Colormap is not supported.");

    return cmap;
}
