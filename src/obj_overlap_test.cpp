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

namespace arma {
    thread_local std::mt19937_64 mt19937_64_instance;
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

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# obj_overlap_test
Detect overlapping 3D objects in a triangular mesh

## Description:
- Tests whether any objects in a triangular mesh overlap by checking for shared volume or intersection.
- Touching faces or edges are not considered overlapping
- Returns the indices (1-based) of all objects that intersect with at least one other object.
- Can optionally output a list of overlap reasons for diagnostic purposes.
- Uses a configurable geometric tolerance to account for numerical precision.
- Allowed datatypes (`dtype`): `float` or `double`.

## Declaration:
```
arma::u32_vec quadriga_lib::obj_overlap_test(
                const arma::Mat<dtype> *mesh,
                const arma::u32_vec *obj_ind,
                std::vector[std::string](std::string) *reason = nullptr,
                dtype tolerance = 0.0005);
```

## Arguments:
- `const arma::Mat<dtype> ***mesh**` (input)<br>
  Triangular mesh geometry. Each row contains 3 vertices flattened as `[X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3]`. Size: `[n_mesh, 9]`.

- `const arma::u32_vec ***obj_ind**` (input)<br>
  Object indices (1-based) that map multiple triangles in `mesh` to objects; Size: `[n_mesh]`;  
  This is an output generated by <a href="#obj_file_read">obj_file_read</a>.

- `std::vector<std::string> ***reason** = nullptr` (optional output)<br>
  Human-readable list of overlap reasons corresponding to each overlapping object. Length: `[n_overlap]`.

- `dtype **tolerance** = 0.0005` (optional input)<br>
  Geometric tolerance (in meters) used to determine intersections. Default: `0.0005` (0.5 mm).

## Returns:
- `arma::u32_vec`: Vector of unique object indices (1-based) that were found to overlap, size `[n_overlap]`.

## Technical Notes:
- Overlap detection includes checks for: Intersecting triangle faces (shared volume), Vertices or edges penetrating another object’s bounding volume.
- The `tolerance` accounts for modeling inaccuracies and numerical instability—small overlaps below this threshold are ignored.
- This function does **not** modify the mesh or attempt to repair overlapping geometry — it only reports it.

## See also:
- <a href="#obj_file_read">obj_file_read</a>
MD!*/

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

    // Direct access to mesh memory
    const dtype *p_mesh = mesh->memptr();

    arma::Mat<dtype> aabb(6ULL, n_obj, arma::fill::none);
    dtype *box = aabb.memptr(); // Object bounding box
    {
        // Initialize bounding box coordinates
        for (arma::uword i = 0ULL; i < aabb.n_elem; ++i)
            box[i] = (i % 2) ? -INFINITY : INFINITY;

        // Get the initial bounding box for the first object
        arma::uword i_obj = 0ULL, i_obj_prev = 0ULL;
        dtype Xmin = box[0ULL], Xmax = box[1ULL],
              Ymin = box[2ULL], Ymax = box[3ULL],
              Zmin = box[4ULL], Zmax = box[5ULL];

        // Calculate the bounding boxes for each object
        for (arma::uword i_mesh = 0ULL; i_mesh < n_mesh; ++i_mesh)
        {
            // Find object index
            for (arma::uword i = 0ULL; i < n_obj; ++i)
            {
                i_obj = i;
                if (p_obj_ids[i] == p_obj_ind[i_mesh])
                    break;
            }
            ++p_faces_per_object[i_obj];

            // Swap bounding box coordinates
            if (i_obj != i_obj_prev)
            {
                box[6ULL * i_obj_prev] = Xmin, Xmin = box[6ULL * i_obj];
                box[6ULL * i_obj_prev + 1ULL] = Xmax, Xmax = box[6ULL * i_obj + 1ULL];
                box[6ULL * i_obj_prev + 2ULL] = Ymin, Ymin = box[6ULL * i_obj + 2ULL];
                box[6ULL * i_obj_prev + 3ULL] = Ymax, Ymax = box[6ULL * i_obj + 3ULL];
                box[6ULL * i_obj_prev + 4ULL] = Zmin, Zmin = box[6ULL * i_obj + 4ULL];
                box[6ULL * i_obj_prev + 5ULL] = Zmax, Zmax = box[6ULL * i_obj + 5ULL];
            }
            i_obj_prev = i_obj;

            dtype x = p_mesh[i_mesh],
                  y = p_mesh[i_mesh + n_mesh],
                  z = p_mesh[i_mesh + 2ULL * n_mesh];

            Xmin = (x < Xmin) ? x : Xmin, Xmax = (x > Xmax) ? x : Xmax;
            Ymin = (y < Ymin) ? y : Ymin, Ymax = (y > Ymax) ? y : Ymax;
            Zmin = (z < Zmin) ? z : Zmin, Zmax = (z > Zmax) ? z : Zmax;

            x = p_mesh[i_mesh + 3ULL * n_mesh],
            y = p_mesh[i_mesh + 4ULL * n_mesh],
            z = p_mesh[i_mesh + 5ULL * n_mesh];

            Xmin = (x < Xmin) ? x : Xmin, Xmax = (x > Xmax) ? x : Xmax;
            Ymin = (y < Ymin) ? y : Ymin, Ymax = (y > Ymax) ? y : Ymax;
            Zmin = (z < Zmin) ? z : Zmin, Zmax = (z > Zmax) ? z : Zmax;

            x = p_mesh[i_mesh + 6ULL * n_mesh],
            y = p_mesh[i_mesh + 7ULL * n_mesh],
            z = p_mesh[i_mesh + 8ULL * n_mesh];

            Xmin = (x < Xmin) ? x : Xmin, Xmax = (x > Xmax) ? x : Xmax;
            Ymin = (y < Ymin) ? y : Ymin, Ymax = (y > Ymax) ? y : Ymax;
            Zmin = (z < Zmin) ? z : Zmin, Zmax = (z > Zmax) ? z : Zmax;
        }

        // Write last update to memory
        box[6ULL * i_obj_prev] = Xmin;
        box[6ULL * i_obj_prev + 1ULL] = Xmax;
        box[6ULL * i_obj_prev + 2ULL] = Ymin;
        box[6ULL * i_obj_prev + 3ULL] = Ymax;
        box[6ULL * i_obj_prev + 4ULL] = Zmin;
        box[6ULL * i_obj_prev + 5ULL] = Zmax;
    }

    // Split the mesh into individual objects
    std::vector<arma::Mat<dtype>> obj_faces(n_obj); // Mesh split into individual objects
    std::vector<arma::Mat<dtype>> obj_edges(n_obj);

#pragma omp parallel for
    for (int i_obj_int = 0; i_obj_int < (int)n_obj; ++i_obj_int)
    {
        arma::uword i_obj = (arma::uword)i_obj_int;

        arma::uword n_faces = p_faces_per_object[i_obj];
        arma::uword n_edges = 3ULL * n_faces;
        obj_faces[i_obj] = arma::Mat<dtype>(18ULL, n_faces, arma::fill::none);
        obj_edges[i_obj] = arma::Mat<dtype>(6ULL, n_edges, arma::fill::none);

        dtype *p_faces = obj_faces[i_obj].memptr();
        dtype *p_edges = obj_edges[i_obj].memptr();

        arma::uword i_face = 0ULL;
        unsigned obj_id = p_obj_ids[i_obj];

        for (arma::uword i_mesh = 0ULL; i_mesh < n_mesh; ++i_mesh)
            if (p_obj_ind[i_mesh] == obj_id)
            {
                arma::uword o_face = 18ULL * i_face;

                // Vertex 1
                dtype x0 = p_mesh[i_mesh];
                dtype y0 = p_mesh[i_mesh + n_mesh];
                dtype z0 = p_mesh[i_mesh + 2ULL * n_mesh];

                p_faces[o_face] = x0;
                p_faces[o_face + 1ULL] = y0;
                p_faces[o_face + 2ULL] = z0;

                p_edges[6ULL * i_face] = x0;
                p_edges[6ULL * i_face + 1ULL] = y0;
                p_edges[6ULL * i_face + 2ULL] = z0;

                p_edges[6ULL * (i_face + 2ULL * n_faces) + 3ULL] = x0;
                p_edges[6ULL * (i_face + 2ULL * n_faces) + 4ULL] = y0;
                p_edges[6ULL * (i_face + 2ULL * n_faces) + 5ULL] = z0;

                // Vertex 2
                dtype x1 = p_mesh[i_mesh + 3ULL * n_mesh];
                dtype y1 = p_mesh[i_mesh + 4ULL * n_mesh];
                dtype z1 = p_mesh[i_mesh + 5ULL * n_mesh];

                p_faces[o_face + 3ULL] = x1;
                p_faces[o_face + 4ULL] = y1;
                p_faces[o_face + 5ULL] = z1;

                p_edges[6ULL * (i_face + n_faces)] = x1;
                p_edges[6ULL * (i_face + n_faces) + 1ULL] = y1;
                p_edges[6ULL * (i_face + n_faces) + 2ULL] = z1;

                p_edges[6ULL * i_face + 3ULL] = x1;
                p_edges[6ULL * i_face + 4ULL] = y1;
                p_edges[6ULL * i_face + 5ULL] = z1;

                // Vertex 3
                dtype x2 = p_mesh[i_mesh + 6ULL * n_mesh];
                dtype y2 = p_mesh[i_mesh + 7ULL * n_mesh];
                dtype z2 = p_mesh[i_mesh + 8ULL * n_mesh];

                p_faces[o_face + 6ULL] = x2;
                p_faces[o_face + 7ULL] = y2;
                p_faces[o_face + 8ULL] = z2;

                p_edges[6ULL * (i_face + 2ULL * n_faces)] = x2;
                p_edges[6ULL * (i_face + 2ULL * n_faces) + 1ULL] = y2;
                p_edges[6ULL * (i_face + 2ULL * n_faces) + 2ULL] = z2;

                p_edges[6ULL * (i_face + n_faces) + 3ULL] = x2;
                p_edges[6ULL * (i_face + n_faces) + 4ULL] = y2;
                p_edges[6ULL * (i_face + n_faces) + 5ULL] = z2;

                // Calculate Edges E1 and E2
                x1 -= x0, y1 -= y0, z1 -= z0;
                x2 -= x0, y2 -= y0, z2 -= z0;

                // Calculate normal vector N - store on X2
                crossp(x1, y1, z1, x2, y2, z2, &x2, &y2, &z2, true);
                p_faces[o_face + 9ULL] = x2;
                p_faces[o_face + 10ULL] = y2;
                p_faces[o_face + 11ULL] = z2;

                // P0 -> P1 (X1) as a direction for U
                dtype len = (dtype)1.0 / std::sqrt(x1 * x1 + y1 * y1 + z1 * z1);
                x1 *= len, y1 *= len, z1 *= len;
                p_faces[o_face + 12ULL] = x1;
                p_faces[o_face + 13ULL] = y1;
                p_faces[o_face + 14ULL] = z1;

                // V = N x U, store result in X2
                crossp(x2, y2, z2, x1, y1, z1, &x2, &y2, &z2, true);
                p_faces[o_face + 15ULL] = x2;
                p_faces[o_face + 16ULL] = y2;
                p_faces[o_face + 17ULL] = z2;

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

    // Performance is better for randomized processing if large objects are clumped together
    arma::uvec rand_obj_indices = arma::regspace<arma::uvec>(0ULL, n_obj - 1ULL);
    rand_obj_indices = arma::shuffle(rand_obj_indices);

    // Pairwise tests
#pragma omp parallel for
    for (int i_obj_1_int = 0; i_obj_1_int < (int)n_obj; ++i_obj_1_int)
    {
        arma::uword i_obj_1 = rand_obj_indices(i_obj_1_int);

        arma::uword n_faces_1 = p_faces_per_object[i_obj_1];
        dtype *p_obj_1 = obj_faces[i_obj_1].memptr();
        dtype ax_low = box[6ULL * i_obj_1] - eps, ax_high = box[6ULL * i_obj_1 + 1ULL] + eps,
              ay_low = box[6ULL * i_obj_1 + 2ULL] - eps, ay_high = box[6ULL * i_obj_1 + 3ULL] + eps,
              az_low = box[6ULL * i_obj_1 + 4ULL] - eps, az_high = box[6ULL * i_obj_1 + 5ULL] + eps;

        for (arma::uword i_obj_2 = 0ULL; i_obj_2 < n_obj; ++i_obj_2)
        {
            if (i_obj_1 == i_obj_2)
                continue;

            // Skip if both objects are already marked as intersecting
            if (intersecting_objects[i_obj_1] && intersecting_objects[i_obj_2])
                continue;

            arma::uword n_faces_2 = p_faces_per_object[i_obj_2];
            dtype *p_obj_2 = obj_faces[i_obj_2].memptr();
            dtype bx_low = box[6ULL * i_obj_2] - eps, bx_high = box[6ULL * i_obj_2 + 1ULL] + eps,
                  by_low = box[6ULL * i_obj_2 + 2ULL] - eps, by_high = box[6ULL * i_obj_2 + 3ULL] + eps,
                  bz_low = box[6ULL * i_obj_2 + 4ULL] - eps, bz_high = box[6ULL * i_obj_2 + 5ULL] + eps;

            // Test if bounding boxes do not overlap
            if (!(ax_high >= bx_low && ax_low <= bx_high &&
                  ay_high >= by_low && ay_low <= by_high &&
                  az_high >= bz_low && az_low <= bz_high))
                continue;

            // Test if all vertices are identical (duplicate objects)
            bool test_condition = n_faces_1 == n_faces_2;
            if (test_condition)
                for (arma::uword i_face = 0ULL; i_face < n_faces_1; ++i_face)
                {
                    arma::uword offset = i_face * 18ULL;
                    for (arma::uword i_val = 0ULL; i_val < 9ULL; ++i_val)
                        if (std::abs(p_obj_1[offset + i_val] - p_obj_2[offset + i_val]) > tolerance)
                        {
                            test_condition = false;
                            break;
                        }
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

            // Test if edges of OBJ-1 intersect with faces of OBJ-2
            std::string findings;
            bool nothing_to_report_so_far = true;

            // Get pointers to data
            arma::uword n_faces = n_faces_1;
            arma::uword n_edges = 3ULL * n_faces_2;

            arma::Mat<dtype> *faces = &obj_faces[i_obj_1];
            arma::Mat<dtype> *edges = &obj_edges[i_obj_2];
            arma::Mat<dtype> *edge_faces = &obj_faces[i_obj_2];

            // Test if edges intersect with faces (3D test)
            arma::u32_vec hit; // Face hit indicator, 1-based, 0 = no hit
            quadriga_lib::ray_triangle_intersect<dtype>(edges, nullptr, faces, nullptr, nullptr, nullptr, &hit, nullptr, nullptr, true);

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
            dtype *p_edges = edges->memptr();
            dtype *p_edge_faces = edge_faces->memptr();

            // Iterate through all faces of OBJ-1
            for (arma::uword i_face = 0ULL; i_face < n_faces; ++i_face)
            {
                arma::uword o_face = i_face * 18ULL;

                // Face vertices
                dtype x0 = p_faces[o_face],
                      y0 = p_faces[o_face + 1ULL],
                      z0 = p_faces[o_face + 2ULL];

                dtype x1 = p_faces[o_face + 3ULL],
                      y1 = p_faces[o_face + 4ULL],
                      z1 = p_faces[o_face + 5ULL];

                dtype x2 = p_faces[o_face + 6ULL],
                      y2 = p_faces[o_face + 7ULL],
                      z2 = p_faces[o_face + 8ULL];

                // Face normal vectors
                dtype Nx = p_faces[o_face + 9ULL],
                      Ny = p_faces[o_face + 10ULL],
                      Nz = p_faces[o_face + 11ULL];

                dtype Ux = p_faces[o_face + 12ULL],
                      Uy = p_faces[o_face + 13ULL],
                      Uz = p_faces[o_face + 14ULL];

                dtype Vx = p_faces[o_face + 15ULL],
                      Vy = p_faces[o_face + 16ULL],
                      Vz = p_faces[o_face + 17ULL];

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
                    arma::uword o_edge = i_edge * 6ULL;

                    // Edge start and end points
                    dtype ox = p_edges[o_edge],
                          oy = p_edges[o_edge + 1ULL],
                          oz = p_edges[o_edge + 2ULL];

                    dtype dx = p_edges[o_edge + 3ULL],
                          dy = p_edges[o_edge + 4ULL],
                          dz = p_edges[o_edge + 5ULL];

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
                        arma::uword o_edge_face = 18ULL * (i_edge % (n_edges / 3ULL));
                        dtype Mx = p_edge_faces[o_edge_face + 9ULL],
                              My = p_edge_faces[o_edge_face + 10ULL],
                              Mz = p_edge_faces[o_edge_face + 11ULL];

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
                            oss << "2D Intersect: OBJ-IDs (" << p_obj_ids[i_obj_1] << "," << p_obj_ids[i_obj_2] << ")";

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
                        oss << "3D Intersect: OBJ-IDs (" << p_obj_ids[i_obj_1] << "," << p_obj_ids[i_obj_2] << ")";

                        unsigned i_face = p_hit[i_edge] - 2U;
                        unsigned o_face = 18U * i_face;

                        dtype x0 = p_faces[o_face],
                              y0 = p_faces[o_face + 1ULL],
                              z0 = p_faces[o_face + 2ULL];

                        dtype x1 = p_faces[o_face + 3ULL],
                              y1 = p_faces[o_face + 4ULL],
                              z1 = p_faces[o_face + 5ULL];

                        dtype x2 = p_faces[o_face + 6ULL],
                              y2 = p_faces[o_face + 7ULL],
                              z2 = p_faces[o_face + 8ULL];

                        oss << " @ F:" << i_face << " {(" << x0 << "," << y0 << "," << z0 << ")"
                            << ",(" << x1 << "," << y1 << "," << z1 << ")"
                            << ",(" << x2 << "," << y2 << "," << z2 << ")}";

                        arma::uword o_edge = 6ULL * i_edge;

                        dtype ox = p_edges[o_edge],
                              oy = p_edges[o_edge + 1ULL],
                              oz = p_edges[o_edge + 2ULL];

                        dtype dx = p_edges[o_edge + 3ULL],
                              dy = p_edges[o_edge + 4ULL],
                              dz = p_edges[o_edge + 5ULL];

                        oss << " vs. E:" << i_edge << "(" << i_edge % n_faces << ") {(" << ox << "," << oy << "," << oz << ")"
                            << ",(" << dx << "," << dy << "," << dz << ")}";

                        findings = oss.str();
                    }
                    if (test_condition && !nothing_to_report_so_far)
                        break;
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
