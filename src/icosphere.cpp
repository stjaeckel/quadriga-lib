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
