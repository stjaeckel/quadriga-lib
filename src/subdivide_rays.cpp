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

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# subdivide_rays
Subdivide ray beams into four smaller sub-beams

## Description:
- Subdivides each ray beam (defined by a triangular wavefront) into four new beams with adjusted origin, shape, and direction.
- Supports input in Spherical or Cartesian direction format.
- When `dest` is not provided, the corresponding output `destN` is omitted.
- Useful for hierarchical ray tracing or angular resolution refinement.
- Allowed datatypes (`dtype`): `float` or `double`.

## Declaration:
```
arma::uword quadriga_lib::subdivide_rays(
                const arma::Mat<dtype> *orig,
                const arma::Mat<dtype> *trivec,
                const arma::Mat<dtype> *tridir,
                const arma::Mat<dtype> *dest = nullptr,
                arma::Mat<dtype> *origN = nullptr,
                arma::Mat<dtype> *trivecN = nullptr,
                arma::Mat<dtype> *tridirN = nullptr,
                arma::Mat<dtype> *destN = nullptr,
                const arma::u32_vec *index = nullptr,
                const double ray_offset = 0.0);
```

## Arguments:
- `const arma::Mat<dtype> ***orig**` (input)<br>
  Ray origin points in global coordinate system (GCS).
  Size: `[n_ray, 3]`.

- `const arma::Mat<dtype> ***trivec**` (input)<br>
  Vectors pointing from the ray origin to the three triangle vertices.
  Size: `[n_ray, 9]`, order: `[x1 y1 z1 x2 y2 z2 x3 y3 z3]`.

- `const arma::Mat<dtype> ***tridir**` (input)<br>
  Directions of the three vertex-rays.
  Format can be Spherical `[n_ray, 6]` as `[v1az v1el v2az v2el v3az v3el]`,
  or Cartesian `[n_ray, 9]` as `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`.

- `const arma::Mat<dtype> ***dest** = nullptr` (input)<br>
  Ray destination points. If `nullptr`, the output `destN` will remain empty.
  Size: `[n_ray, 3]`.

- `arma::Mat<dtype> ***origN**` (output)<br>
  New ray origins after subdivision.
  Size: `[n_rayN, 3]`.

- `arma::Mat<dtype> ***trivecN**` (output)<br>
  Updated vectors for each subdivided triangle beam.
  Size: `[n_rayN, 9]`.

- `arma::Mat<dtype> ***tridirN**` (output)<br>
  New directions of the subdivided vertex-rays, in the same format as input.
  Size: `[n_rayN, 6]` (spherical) or `[n_rayN, 9]` (Cartesian).

- `arma::Mat<dtype> ***destN**` (output)<br>
  Updated destination points.
  Size: `[n_rayN, 3]`, empty if input `dest` was `nullptr`.

- `const arma::u32_vec ***index**` (optional input)<br>
  List of ray indices to be subdivided (0-based). Only the specified rays are subdivided.
  Size: `[n_ind]`.

- `const double **ray_offset** = 0.0` (optional input)<br>
  Offset (in meters) applied to the origin of each subdivided ray along its propagation direction.
  Default: `0.0`.

## Returns:
- `arma::uword  **n_rayN**`<br>
  Number of output rays, typically `4 × n_ray` or `4 × n_ind` if `index` is provided.

## See also:
- <a href="#icosphere">icosphere</a> (for generating beams)
- <a href="#ray_point_intersect">ray_point_intersect</a> (for calculating beam interactions with sampling points)
- <a href="#ray_triangle_intersect">ray_triangle_intersect</a> (for calculating beam interactions with triangles)
MD!*/

template <typename dtype>
arma::uword quadriga_lib::subdivide_rays(const arma::Mat<dtype> *orig, const arma::Mat<dtype> *trivec, const arma::Mat<dtype> *tridir, const arma::Mat<dtype> *dest,
                                         arma::Mat<dtype> *origN, arma::Mat<dtype> *trivecN, arma::Mat<dtype> *tridirN, arma::Mat<dtype> *destN,
                                         const arma::u32_vec *index, const double ray_offset)
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

    const arma::uword n_ray = orig->n_rows; // Number of rays
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

    arma::uword n_ind = 0;
    unsigned *p_ind;
    if (index != nullptr && index->n_elem != 0)
    {
        n_ind = (arma::uword)index->n_elem;
        const unsigned *tmp = index->memptr();

        for (arma::uword i = 0; i < n_ind; ++i)
            if (tmp[i] >= (unsigned)n_ray)
                throw std::invalid_argument("Indices cannot exceed number of rays.");

        p_ind = new unsigned[n_ind];
        std::memcpy(p_ind, tmp, n_ind * sizeof(unsigned));
    }
    else
    {
        n_ind = n_ray;
        p_ind = new unsigned[n_ray];
        for (unsigned i_ray = 0; i_ray < n_ray_u; ++i_ray)
            p_ind[i_ray] = i_ray;
    }

    // Number of rays in the output
    arma::uword n_rayN = 4 * n_ind;

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
    for (arma::uword i_ind = 0; i_ind < n_ind; ++i_ind)
    {
        arma::uword i_ray = p_ind[i_ind];

        // Load beam origin
        double Ox = (double)p_orig[i_ray],
               Oy = (double)p_orig[i_ray + n_ray],
               Oz = (double)p_orig[i_ray + 2 * n_ray];

        // Load destination and calculate the length from orig to dest
        double length = NAN;
        if (p_dest != nullptr)
        {
            double Ux = (double)p_dest[i_ray] - Ox;
            double Uy = (double)p_dest[i_ray + n_ray] - Oy;
            double Uz = (double)p_dest[i_ray + 2 * n_ray] - Oz;
            length = std::sqrt(Ux * Ux + Uy * Uy + Uz * Uz);
        }

        // Load the 3 beam vertices
        double W1x = Ox + (double)p_trivec[i_ray],
               W1y = Oy + (double)p_trivec[i_ray + n_ray],
               W1z = Oz + (double)p_trivec[i_ray + 2 * n_ray];

        double W2x = Ox + (double)p_trivec[i_ray + 3 * n_ray],
               W2y = Oy + (double)p_trivec[i_ray + 4 * n_ray],
               W2z = Oz + (double)p_trivec[i_ray + 5 * n_ray];

        double W3x = Ox + (double)p_trivec[i_ray + 6 * n_ray],
               W3y = Oy + (double)p_trivec[i_ray + 7 * n_ray],
               W3z = Oz + (double)p_trivec[i_ray + 8 * n_ray];

        // Calculate the 3 additional vertices
        double W12x = 0.5 * (W1x + W2x), W12y = 0.5 * (W1y + W2y), W12z = 0.5 * (W1z + W2z);
        double W13x = 0.5 * (W1x + W3x), W13y = 0.5 * (W1y + W3y), W13z = 0.5 * (W1z + W3z);
        double W23x = 0.5 * (W2x + W3x), W23y = 0.5 * (W2y + W3y), W23z = 0.5 * (W2z + W3z);

        // Calculate the direction vectors at the vertices
        double D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z, scl;
        if (cartesian_format)
        {
            D1x = (double)p_tridir[i_ray];
            D1y = (double)p_tridir[i_ray + n_ray];
            D1z = (double)p_tridir[i_ray + 2 * n_ray];

            scl = D1x * D1x + D1y * D1y + D1z * D1z;
            if (std::abs(scl - 1.0) > 2.0e-7) // Normalize
                scl = 1.0 / std::sqrt(scl), D1x *= scl, D1y *= scl, D1z *= scl;

            D2x = (double)p_tridir[i_ray + 3 * n_ray];
            D2y = (double)p_tridir[i_ray + 4 * n_ray];
            D2z = (double)p_tridir[i_ray + 5 * n_ray];

            scl = D2x * D2x + D2y * D2y + D2z * D2z;
            if (std::abs(scl - 1.0) > 2.0e-7) // Normalize
                scl = 1.0 / std::sqrt(scl), D2x *= scl, D2y *= scl, D2z *= scl;

            D3x = (double)p_tridir[i_ray + 6 * n_ray];
            D3y = (double)p_tridir[i_ray + 7 * n_ray];
            D3z = (double)p_tridir[i_ray + 8 * n_ray];

            scl = D3x * D3x + D3y * D3y + D3z * D3z;
            if (std::abs(scl - 1.0) > 2.0e-7) // Normalize
                scl = 1.0 / std::sqrt(scl), D3x *= scl, D3y *= scl, D3z *= scl;
        }
        else // Spherical format
        {
            double az = (double)p_tridir[i_ray],
                   el = (double)p_tridir[i_ray + n_ray];

            scl = std::cos(el);
            D1x = std::cos(az) * scl, D1y = std::sin(az) * scl, D1z = std::sin(el);

            az = (double)p_tridir[i_ray + 2 * n_ray];
            el = (double)p_tridir[i_ray + 3 * n_ray];

            scl = std::cos(el);
            D2x = std::cos(az) * scl, D2y = std::sin(az) * scl, D2z = std::sin(el);

            az = (double)p_tridir[i_ray + 4 * n_ray];
            el = (double)p_tridir[i_ray + 5 * n_ray];

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
        for (arma::uword i_sub = 0; i_sub < 4; ++i_sub)
        {
            // Index of the ray in the output
            arma::uword i_rayN = 4 * i_ind + i_sub;

            double w1x, w1y, w1z, w2x, w2y, w2z, w3x, w3y, w3z;
            double d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z;

            if (i_sub == 0)
            {
                w1x = W1x, w1y = W1y, w1z = W1z, w2x = W12x, w2y = W12y, w2z = W12z, w3x = W13x, w3y = W13y, w3z = W13z;
                d1x = D1x, d1y = D1y, d1z = D1z, d2x = D12x, d2y = D12y, d2z = D12z, d3x = D13x, d3y = D13y, d3z = D13z;

                if (p_tridirN != nullptr && !cartesian_format)
                {
                    p_tridirN[i_rayN] = p_tridir[i_ray];
                    p_tridirN[i_rayN + n_rayN] = p_tridir[i_ray + n_ray];
                    p_tridirN[i_rayN + 2 * n_rayN] = az12;
                    p_tridirN[i_rayN + 3 * n_rayN] = el12;
                    p_tridirN[i_rayN + 4 * n_rayN] = az13;
                    p_tridirN[i_rayN + 5 * n_rayN] = el13;
                }
            }
            else if (i_sub == 1)
            {
                w1x = W13x, w1y = W13y, w1z = W13z, w2x = W12x, w2y = W12y, w2z = W12z, w3x = W23x, w3y = W23y, w3z = W23z;
                d1x = D13x, d1y = D13y, d1z = D13z, d2x = D12x, d2y = D12y, d2z = D12z, d3x = D23x, d3y = D23y, d3z = D23z;

                if (p_tridirN != nullptr && !cartesian_format)
                {
                    p_tridirN[i_rayN] = az13;
                    p_tridirN[i_rayN + n_rayN] = el13;
                    p_tridirN[i_rayN + 2 * n_rayN] = az12;
                    p_tridirN[i_rayN + 3 * n_rayN] = el12;
                    p_tridirN[i_rayN + 4 * n_rayN] = az23;
                    p_tridirN[i_rayN + 5 * n_rayN] = el23;
                }
            }
            else if (i_sub == 2)
            {
                w1x = W13x, w1y = W13y, w1z = W13z, w2x = W23x, w2y = W23y, w2z = W23z, w3x = W3x, w3y = W3y, w3z = W3z;
                d1x = D13x, d1y = D13y, d1z = D13z, d2x = D23x, d2y = D23y, d2z = D23z, d3x = D3x, d3y = D3y, d3z = D3z;

                if (p_tridirN != nullptr && !cartesian_format)
                {
                    p_tridirN[i_rayN] = az13;
                    p_tridirN[i_rayN + n_rayN] = el13;
                    p_tridirN[i_rayN + 2 * n_rayN] = az23;
                    p_tridirN[i_rayN + 3 * n_rayN] = el23;
                    p_tridirN[i_rayN + 4 * n_rayN] = p_tridir[i_ray + 4 * n_ray];
                    p_tridirN[i_rayN + 5 * n_rayN] = p_tridir[i_ray + 5 * n_ray];
                }
            }
            else if (i_sub == 3)
            {
                w1x = W12x, w1y = W12y, w1z = W12z, w2x = W2x, w2y = W2y, w2z = W2z, w3x = W23x, w3y = W23y, w3z = W23z;
                d1x = D12x, d1y = D12y, d1z = D12z, d2x = D2x, d2y = D2y, d2z = D2z, d3x = D23x, d3y = D23y, d3z = D23z;

                if (p_tridirN != nullptr && !cartesian_format)
                {
                    p_tridirN[i_rayN] = az12;
                    p_tridirN[i_rayN + n_rayN] = el12;
                    p_tridirN[i_rayN + 2 * n_rayN] = p_tridir[i_ray + 2 * n_ray];
                    p_tridirN[i_rayN + 3 * n_rayN] = p_tridir[i_ray + 3 * n_ray];
                    p_tridirN[i_rayN + 4 * n_rayN] = az23;
                    p_tridirN[i_rayN + 5 * n_rayN] = el23;
                }
            }

            // Write "tridir" for Cartesian format
            if (p_tridirN != nullptr && cartesian_format)
            {
                p_tridirN[i_rayN] = (dtype)d1x;
                p_tridirN[i_rayN + n_rayN] = (dtype)d1y;
                p_tridirN[i_rayN + 2 * n_rayN] = (dtype)d1z;
                p_tridirN[i_rayN + 3 * n_rayN] = (dtype)d2x;
                p_tridirN[i_rayN + 4 * n_rayN] = (dtype)d2y;
                p_tridirN[i_rayN + 5 * n_rayN] = (dtype)d2z;
                p_tridirN[i_rayN + 6 * n_rayN] = (dtype)d3x;
                p_tridirN[i_rayN + 7 * n_rayN] = (dtype)d3y;
                p_tridirN[i_rayN + 8 * n_rayN] = (dtype)d3z;
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
                p_destN[i_rayN + n_rayN] = (dtype)ty;
                p_destN[i_rayN + 2 * n_rayN] = (dtype)tz;
            }

            // Calculate new origin point (consider ray offset)
            ox += ray_offset * dx, oy += ray_offset * dy, oz += ray_offset * dz;

            // Write new origin point
            if (p_origN != nullptr)
            {
                p_origN[i_rayN] = (dtype)ox;
                p_origN[i_rayN + n_rayN] = (dtype)oy;
                p_origN[i_rayN + 2 * n_rayN] = (dtype)oz;
            }

            // Write new trivec
            if (p_trivecN != nullptr)
            {
                double tx = w1x - ox, ty = w1y - oy, tz = w1z - oz;

                p_trivecN[i_rayN] = (dtype)tx;
                p_trivecN[i_rayN + n_rayN] = (dtype)ty;
                p_trivecN[i_rayN + 2 * n_rayN] = (dtype)tz;

                tx = w2x - ox, ty = w2y - oy, tz = w2z - oz;

                p_trivecN[i_rayN + 3 * n_rayN] = (dtype)tx;
                p_trivecN[i_rayN + 4 * n_rayN] = (dtype)ty;
                p_trivecN[i_rayN + 5 * n_rayN] = (dtype)tz;

                tx = w3x - ox, ty = w3y - oy, tz = w3z - oz;

                p_trivecN[i_rayN + 6 * n_rayN] = (dtype)tx;
                p_trivecN[i_rayN + 7 * n_rayN] = (dtype)ty;
                p_trivecN[i_rayN + 8 * n_rayN] = (dtype)tz;
            }
        }
    }

    delete[] p_ind;
    return n_rayN;
}

template arma::uword quadriga_lib::subdivide_rays(const arma::Mat<float> *orig, const arma::Mat<float> *trivec, const arma::Mat<float> *tridir, const arma::Mat<float> *dest,
                                                  arma::Mat<float> *origN, arma::Mat<float> *trivecN, arma::Mat<float> *tridirN, arma::Mat<float> *destN,
                                                  const arma::u32_vec *index, const double ray_offset);

template arma::uword quadriga_lib::subdivide_rays(const arma::Mat<double> *orig, const arma::Mat<double> *trivec, const arma::Mat<double> *tridir, const arma::Mat<double> *dest,
                                                  arma::Mat<double> *origN, arma::Mat<double> *trivecN, arma::Mat<double> *tridirN, arma::Mat<double> *destN,
                                                  const arma::u32_vec *index, const double ray_offset);
