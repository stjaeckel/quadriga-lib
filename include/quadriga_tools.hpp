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

#ifndef quadriga_tools_H
#define quadriga_tools_H

#include <armadillo>
#include <string>

typedef unsigned long long int uword;
typedef long long int sword;

namespace quadriga_lib
{
    template <typename dtype> // float or double
    arma::cube calc_rotation_matrix(const arma::Cube<dtype> orientation,
                                    bool invert_y_axis = false, bool transposeR = false);

    template <typename dtype> // float or double
    arma::cube cart2geo(const arma::Cube<dtype> cart);

    // Convert path interaction coordinates into FBS/LBS positions, path length and angles
    // - Interaction coordinates for the LOS path must be all nan
    // - FBS / LBS position of the LOS path is place half way between TX and RX
    // - Size of the output arguments is adjusted if it does not match the required size
    template <typename dtype>                                 // Supported types: float or double
    void coord2path(dtype Tx, dtype Ty, dtype Tz,             // Transmitter position in Cartesian coordinates
                    dtype Rx, dtype Ry, dtype Rz,             // Receiver position in Cartesian coordinates
                    const arma::Cube<dtype> *coord,           // Interaction coordinates, NAN-padded, size [3, n_coord, n_path]
                    arma::Col<dtype> *path_length = nullptr,  // Absolute path length from TX to RX phase center, vector of length [n_path]
                    arma::Mat<dtype> *fbs_pos = nullptr,      // First-bounce scatterer positions, matrix of size [3, n_path]
                    arma::Mat<dtype> *lbs_pos = nullptr,      // Last-bounce scatterer positions, matrix of size [3, n_path]
                    arma::Mat<dtype> *path_angles = nullptr); // Departure and arrival angles {AOD, EOD, AOA, EOA}, matrix of size [n_path, 4]

    template <typename dtype> // float or double
    arma::cube geo2cart(const arma::Mat<dtype> azimuth, const arma::Mat<dtype> elevation, const arma::Mat<dtype> length);

    // Construct a geodesic polyhedron (icosphere), a convex polyhedron made from triangles
    // - Returns the number of faces
    template <typename dtype>                               // Allowed types: float or double
    uword icosphere(uword n_div,                            // Number of sub-segments per edge, results in n_faces = 20 * n_div^2 elements
                    dtype radius,                           // Radius of the icosphere in meters
                    arma::Mat<dtype> *center,               // Pointing vector from the origin to the center of the triangle, matrix of size [no_faces, 3]
                    arma::Col<dtype> *length = nullptr,     // Length of the pointing vector "center" (slightly smaller than 1), vector of length [no_faces]
                    arma::Mat<dtype> *vert = nullptr,       // Vectors pointing from "center" to the vertices of the triangle, matrix of size [no_ray, 9], [x1 y1 z1 x2 y2 z3 x3 y3 z3]
                    arma::Mat<dtype> *direction = nullptr); // Directions of the vertex-rays in rad; matrix of size [no_ray, 6], the values are in the order [ v1az, v1el, v2az, v2el, v3az, v3el ]

    // 2D linear interpolation (returns error message or empty string in case of no error)
    template <typename dtype>                          // Supported types: float or double
    std::string interp(const arma::Cube<dtype> *input, // Input data; size [ ny, nx, ne ], ne = multiple data sets
                       const arma::Col<dtype> *xi,     // x sample points of input; vector length nx
                       const arma::Col<dtype> *yi,     // y sample points of input; vector length ny
                       const arma::Col<dtype> *xo,     // x sample points of output; vector length mx
                       const arma::Col<dtype> *yo,     // y sample points of output; vector length my
                       arma::Cube<dtype> *output);     // Interpolated data; size [ my, mx, ne ]

    // 1D linear interpolation (returns error message or empty string)
    template <typename dtype>                         // Supported types: float or double
    std::string interp(const arma::Mat<dtype> *input, // Input data; size [ nx, ne ], ne = multiple data sets
                       const arma::Col<dtype> *xi,    // x sample points of input; vector length nx
                       const arma::Col<dtype> *xo,    // x sample points of output; vector length mx
                       arma::Mat<dtype> *output);     // Interpolated data; size [ mx, ne ]

    // Read Wavefront .obj file
    // - See: https://en.wikipedia.org/wiki/Wavefront_.obj_file
    // - 3D model must be triangularized
    // - Material properties are encoded in the material name using the "usemtl [material name]" tag
    // - Default material properties are taken from ITU-R P.2040-1, Table 3
    // - Default materials: Concrete, Brick, Plasterboard, Wood, Glass, Chipboard, Metal,
    //                      Ground_dry, Ground_medium, Ground_wet, Vegetation, Water, Ice, IRR_glass
    // - Supported frequency range: 1 - 100 GHz (1 - 10 GHz for Ground materials)
    // - Custom materials can be defined by: "usemtl Name::A:B:C:D:att"
    //          - Real part of relative permittivity "eta = A * fGHz ^ B"
    //          - Conductivity "sigma = C * fGHz ^ D"
    //          - att = Material penetration loss in dB (fixed loss per material interaction)
    // - Unknown materials are treated as Vacuum
    // - Returns number of mesh elements "n_mesh"
    // - Attempts to change the size of the output if it does not have correct size already
    template <typename dtype>                                       // Supported types: float or double
    unsigned obj_file_read(std::string fn,                          // File name
                           arma::Mat<dtype> *mesh = nullptr,        // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                           arma::Mat<dtype> *mtl_prop = nullptr,    // Material properties, Size: [ n_mesh, 5 ]
                           arma::Mat<dtype> *vert_list = nullptr,   // List of vertices found in the OBJ file, Size: [ n_vert, 3 ]
                           arma::Mat<unsigned> *face_ind = nullptr, // Vertex indices matching the corresponding mesh elements, 0-based, Size: [ n_mesh, 3 ]
                           arma::Col<unsigned> *obj_ind = nullptr,  // Object index, 1-based, Size: [ n_mesh ]
                           arma::Col<unsigned> *mtl_ind = nullptr); // Material index, 1-based, Size: [ n_mesh ]

    // Subdivide triangles into smaller triangles
    // - Returns the number of triangles after subdivision
    // - Attempts to change the size of the output if it does not match [n_triangles_out, 9]
    template <typename dtype>                                       // Supported types: float or double
    uword subdivide_triangles(uword n_div,                          // Number of divisions per edge, results in: n_triangles_out = n_triangles_in * n_div^2
                              const arma::Mat<dtype> *triangles_in, // Input, matrix of size [n_triangles_in, 9]
                              arma::Mat<dtype> *triangles_out);     // Output, matrix of size [n_triangles_out, 9]

}

#endif
