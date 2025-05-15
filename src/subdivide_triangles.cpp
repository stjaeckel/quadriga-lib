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
# subdivide_triangles
Subdivide triangles into smaller triangles

## Description:
- Uniformly subdivides each input triangle into `n_div × n_div` smaller triangles.
- Increases spatial resolution for mesh-based processing (e.g., ray tracing or visualization).
- Optional input/output material properties are duplicated across subdivided triangles.
- Allowed datatypes (`dtype`): `float` or `double`.

## Declaration:
```
arma::uword quadriga_lib::subdivide_triangles(
                arma::uword n_div,
                const arma::Mat<dtype> *triangles_in,
                arma::Mat<dtype> *triangles_out,
                const arma::Mat<dtype> *mtl_prop = nullptr,
                arma::Mat<dtype> *mtl_prop_out = nullptr);
```

## Arguments:
- `arma::uword **n_div**` (input)<br>
  Number of subdivisions per triangle edge;
  total output triangles: `n_triangles_out = n_triangles_in × n_div × n_div`.

- `const arma::Mat<dtype> ***triangles_in**` (input)<br>
  Vertices of the triangular mesh in global Cartesian coordinates; each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[n_triangles_in, 9]`

- `arma::Mat<dtype> ***triangles_out**` (output)<br>
  Vertices of the sub-divided mesh in global Cartesian coordinates; Size: `[n_triangles_out, 9]`

- `const arma::Mat<dtype> ***mtl_prop** = nullptr` (optional input)<br>
  Material properties associated for the input triangles; Size: `[n_triangles_in, 5]`.

- `arma::Mat<dtype> ***mtl_prop_out** = nullptr` (optional output)<br>
  Material properties for the subdivided triangles, copied from the parent triangle,
  Size: `[n_triangles_out, 5]`.

## Returns:
- `arma::uword **n_triangles_out**`<br>
  Number of generated triangles (equals `n_triangles_in × n_div × n_div`).
MD!*/

// Subdivide triangles into smaller triangles
template <typename dtype>
arma::uword quadriga_lib::subdivide_triangles(arma::uword n_div, const arma::Mat<dtype> *triangles_in, arma::Mat<dtype> *triangles_out,
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

    arma::uword n_triangles_in = (arma::uword)triangles_in->n_rows;
    arma::uword n_triangles_out = n_triangles_in * n_div * n_div;

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
    arma::uword cnt = 0;                                  // Counter
    dtype stp = dtype(1.0) / dtype(n_div);                // Step size
    const dtype *p_triangles_in = triangles_in->memptr(); // Pointer to input memory
    dtype *p_triangles_out = triangles_out->memptr();     // Pointer to output memory
    const dtype *p_mtl_prop = process_mtl_prop ? mtl_prop->memptr() : nullptr;
    dtype *p_mtl_prop_out = process_mtl_prop ? mtl_prop_out->memptr() : nullptr;

    for (arma::uword n = 0; n < n_triangles_in; ++n)
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

        for (arma::uword u = 0; u < n_div; ++u)
        {
            dtype ful = dtype(u) * stp;
            dtype fuu = ful + stp;

            for (arma::uword v = 0; v < n_div - u; ++v)
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

template arma::uword quadriga_lib::subdivide_triangles(arma::uword n_div, const arma::Mat<float> *triangles_in, arma::Mat<float> *triangles_out,
                                                       const arma::Mat<float> *mtl_prop, arma::Mat<float> *mtl_prop_out);

template arma::uword quadriga_lib::subdivide_triangles(arma::uword n_div, const arma::Mat<double> *triangles_in, arma::Mat<double> *triangles_out,
                                                       const arma::Mat<double> *mtl_prop, arma::Mat<double> *mtl_prop_out);
