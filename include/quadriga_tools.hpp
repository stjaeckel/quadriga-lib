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

#ifndef quadriga_tools_H
#define quadriga_tools_H

#include <armadillo>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <filesystem>

// If arma::uword and size_t are not the same width (e.g. 64 bit), the compiler will throw an error here
// This allows the use of "uword", "size_t" and "unsigned long long" interchangeably
// This requires a 64 bit platform, but will compile on Linux, Windows and macOS
static_assert(sizeof(arma::uword) == sizeof(unsigned long long), "arma::uword and unsigned long long have different sizes");
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t and unsigned long long have different sizes");

namespace quadriga_lib
{

    // ---- Miscellaneous tools ----

    // Calculates the rotation matrix from Euler angles
    // - Returns: Coefficients of the rotation matrix in column-major ordering, Size: [9, n_row, n_col]
    // - Internal calculations are done in double precision, even if the <dtype> is <float>
    template <typename dtype>
    arma::Cube<dtype> calc_rotation_matrix(const arma::Cube<dtype> &orientation, // Orientation vectors (Euler rotations) in [rad], Size [3, n_row, n_col]
                                           bool invert_y_axis = false,           // Inverts the y-axis
                                           bool transposeR = false);             // Returns the transpose of R instead of R

    template <typename dtype>
    arma::Mat<dtype> calc_rotation_matrix(const arma::Mat<dtype> &orientation, bool invert_y_axis = false, bool transposeR = false);

    template <typename dtype>
    arma::Col<dtype> calc_rotation_matrix(const arma::Col<dtype> &orientation, bool invert_y_axis = false, bool transposeR = false);

    // Transform Cartesian (x,y,z) coordinates to Geographic (az, el, length) coordinates
    // - Input: Cartesian coordinates, size [3, n_row, n_col]
    // - Output: Geographic coordinates, size [n_row, n_col, 3]
    template <typename dtype>
    void cart2geo(const arma::Cube<dtype> &cart, arma::Cube<dtype> &geo);

    template <typename dtype>
    void cart2geo(const arma::Mat<dtype> &cart, arma::Mat<dtype> &geo); // Input: [3, n_row], Output: [n_row, 3]

    template <typename dtype>
    void cart2geo(const arma::Col<dtype> &cart, arma::Col<dtype> &geo); // Input / Output: [3]

    template <typename arma_type>
    arma_type cart2geo(const arma_type &cart);

    // Generate colormap
    // - Returns a 64 x 3 matrix of unsigned chars
    // - Supported colormaps: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer
    arma::uchar_mat colormap(std::string map, bool high_res = false);

    // Combine real and imaginary parts to complex types, cast to double
    template <typename dtype>
    void complex_cast(const arma::Mat<dtype> &real, // Matrix (real part, float or double)
                      const arma::Mat<dtype> &imag, // Matrix (imaginary part, float or double)
                      arma::cx_mat &complex);       // Output: Complex matrix, double

    template <typename dtype>
    void complex_cast(const arma::Cube<dtype> &real, // Cube (real part, float or double)
                      const arma::Cube<dtype> &imag, // Cube (imaginary part, float or double)
                      arma::cx_cube &complex);       // Output: Complex cube, double

    template <typename dtype>
    void complex_cast(const arma::cx_mat &complex, // Complex matrix, double
                      arma::Mat<dtype> &real,      // Output: Matrix (real part, float or double)
                      arma::Mat<dtype> &imag);     // Output: Matrix (imaginary part, float or double)

    template <typename dtype>
    void complex_cast(const arma::cx_cube &complex, // Complex cube, double
                      arma::Cube<dtype> &real,      // Output: Cube (real part, float or double)
                      arma::Cube<dtype> &imag);     // Output: Cube (imaginary part, float or double)

    // Transform Geographic (az, el, length) to Cartesian (x,y,z) coordinates coordinates
    // - Returns: Cartesian coordinates, Size [3, n_row, n_col]
    template <typename dtype>
    arma::Cube<dtype> geo2cart(const arma::Mat<dtype> &azimuth,      // Azimuth angles, Size [n_row, n_col]
                               const arma::Mat<dtype> &elevation,    // Elevation angles, Size [n_row, n_col]
                               const arma::Mat<dtype> &length = {}); // Radius, Size [n_row, n_col] or [0,0] for unit length

    // 2D linear interpolation of multiple data sets
    // - Output object is passed as a reference (its data will be overwritten)
    // - If output is not initialized to size [ my, mx, me ], it will be resized, invalidating data pointers
    template <typename dtype>
    void interp_2D(const arma::Cube<dtype> &input, // Input data; size [ ny, nx, ne ], ne = multiple data sets
                   const arma::Col<dtype> &xi,     // x sample points of input; vector length nx
                   const arma::Col<dtype> &yi,     // y sample points of input; vector length ny
                   const arma::Col<dtype> &xo,     // x sample points of output; vector length mx
                   const arma::Col<dtype> &yo,     // y sample points of output; vector length my
                   arma::Cube<dtype> &output);     // Interpolated data, size [ my, mx, me ]

    // 2D linear interpolation of multiple data sets
    // - Returns interpolated data, size [ my, mx, me ]
    template <typename dtype>
    arma::Cube<dtype> interp_2D(const arma::Cube<dtype> &input, // Input data; size [ ny, nx, ne ], ne = multiple data sets
                                const arma::Col<dtype> &xi,     // x sample points of input; vector length nx
                                const arma::Col<dtype> &yi,     // y sample points of input; vector length ny
                                const arma::Col<dtype> &xo,     // x sample points of output; vector length mx
                                const arma::Col<dtype> &yo);    // y sample points of output; vector length my

    // 2D linear interpolation of a single data set
    // - Returns interpolated data, size [ my, mx ]
    template <typename dtype>
    arma::Mat<dtype> interp_2D(const arma::Mat<dtype> &input,                           // Input data; size [ ny, nx ]
                               const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,  // x/y input sample points
                               const arma::Col<dtype> &xo, const arma::Col<dtype> &yo); // x/y output sample points

    // 1D linear interpolation of multiple data sets
    // - Returns interpolated data, size [ mx, ne ]
    template <typename dtype>
    arma::Mat<dtype> interp_1D(const arma::Mat<dtype> &input, // Input data; size [ nx, ne ], ne = multiple data sets
                               const arma::Col<dtype> &xi,    // Input sample points, vector length nx
                               const arma::Col<dtype> &xo);   // Output sample points, vector length mx

    // 1D linear interpolation of single data set
    // - Returns interpolated data, length mx
    template <typename dtype>
    arma::Col<dtype> interp_1D(const arma::Col<dtype> &input, // Input data vector, vector length nx
                               const arma::Col<dtype> &xi,    // Input sample points, vector length nx
                               const arma::Col<dtype> &xo);   // Output sample points, vector length mx

    // Write data to PNG file
    template <typename dtype>                    // Types: float, double
    void write_png(const arma::Mat<dtype> &data, // Data matrix
                   std::string fn,               // Filename of the PNG file, string
                   std::string colormap = "jet", // Colormap
                   dtype min_val = NAN,          // Minimum value, when passing NAN, minimum in data is used
                   dtype max_val = NAN,          // Maximum value, when passing NAN, maximum data is used
                   bool log_transform = false);  // Transform data to log-domain (10*log10(data))

    // ---- Site-Specific Simulation Tools ----

    // Calculate diffraction gain for multiple transmit and receive positions
    // - For faster processing, the mesh can be split into sub-meshes. In this case, the algorithm can
    //   efficiently skip entire sub-meshed if no interaction with any of its componenets is possible
    //   (i.e. the path does not insect its bounding-box).
    // - The function "quadriga_lib::triangle_mesh_segmentation" can be used to split the mesh and
    //   calculate the optional "sub_mesh_index".
    template <typename dtype>                                                  // Supported types: float or double
    void calc_diffraction_gain(const arma::Mat<dtype> *orig,                   // Origin points, Size: [ n_pos, 3 ]
                               const arma::Mat<dtype> *dest,                   // Destination points, Size: [ n_pos, 3 ]
                               const arma::Mat<dtype> *mesh,                   // Vertices of the triangular mesh; Size: [ no_mesh, 9 ]
                               const arma::Mat<dtype> *mtl_prop,               // Material properties; Size: [ no_mesh, 5 ]
                               dtype center_frequency,                         // Center frequency in [Hz]
                               int lod = 2,                                    // Level of detail, scalar values 0-6
                               arma::Col<dtype> *gain = nullptr,               // Diffraction gain; linear scale; Size: [ n_pos ]
                               arma::Cube<dtype> *coord = nullptr,             // Approximate coordinates of the diffracted path; [ 3, n_seg-1, n_pos ]
                               int verbose = 0,                                // Verbosity level
                               const arma::u32_vec *sub_mesh_index = nullptr); // Sub-mesh index, 0-based, (optional input), Length: [ n_sub ]

    // Convert path interaction coordinates into FBS/LBS positions, path length and angles
    // - FBS / LBS position of the LOS path is placed half way between TX and RX
    // - Size of the output arguments is adjusted if it does not match the required size
    template <typename dtype>                                            // Supported types: float or double
    void coord2path(dtype Tx, dtype Ty, dtype Tz,                        // Transmitter position in Cartesian coordinates
                    dtype Rx, dtype Ry, dtype Rz,                        // Receiver position in Cartesian coordinates
                    const arma::u32_vec *no_interact,                    // Number interaction points of a path with the environment, 0 = LOS, vector of length [n_path]
                    const arma::Mat<dtype> *interact_coord,              // Interaction coordinates of paths with the environment, matrix of size [3, sum(no_interact)]
                    arma::Col<dtype> *path_length = nullptr,             // Absolute path length from TX to RX phase center, vector of length [n_path]
                    arma::Mat<dtype> *fbs_pos = nullptr,                 // First-bounce scatterer positions, matrix of size [3, n_path]
                    arma::Mat<dtype> *lbs_pos = nullptr,                 // Last-bounce scatterer positions, matrix of size [3, n_path]
                    arma::Mat<dtype> *path_angles = nullptr,             // Departure and arrival angles {AOD, EOD, AOA, EOA}, matrix of size [n_path, 4]
                    std::vector<arma::Mat<dtype>> *path_coord = nullptr, // Interaction coordinates, vector (n_path) of matrices of size [3, n_interact + 2]
                    bool reverse_path = false);                          // Option to reverse the path (swap TX and RX positions), including TX and RS positions

    // Combine path interaction coordinates for Intelligent Reflective Surfaces (IRS)
    // - Requires 2 channel segments: (1) TX -> IRS and (2) IRS -> RX
    // - Generates output for n_path_irs paths where n_path_irs <= n_path_1 * n_path_2
    // - Optional input 'active_path' selects a subset of paths generated by 'channel::get_channels_irs'
    // - The reverse_segment options only reverse the interaction coordinates for the segment. TX / IRS / RX positions are not reversed.
    template <typename dtype>                                               // Supported types: float or double
    void combine_irs_coord(dtype Ix, dtype Iy, dtype Iz,                    // IRS position in Cartesian coordinates
                           const arma::u32_vec *no_interact_1,              // Number interaction points for segment 1, 0 = LOS, vector of length [n_path_1]
                           const arma::Mat<dtype> *interact_coord_1,        // Interaction coordinates for segment 1, matrix of size [3, sum(no_interact_1)]
                           const arma::u32_vec *no_interact_2,              // Number interaction points for segment 2, 0 = LOS, vector of length [n_path_2]
                           const arma::Mat<dtype> *interact_coord_2,        // Interaction coordinates for segment 2, matrix of size [3, sum(no_interact_2)]
                           arma::u32_vec *no_interact,                      // Output: Combined number of interaction coordinates, vector of length [n_path_irs]
                           arma::Mat<dtype> *interact_coord,                // Output: Combined interaction coordinates, matrix of size [3, sum(no_interact_irs)]
                           bool reverse_segment_1 = false,                  // Option to reverse interact_coord for segment 1 (TX and IRS positions swapped)
                           bool reverse_segment_2 = false,                  // Option to reverse interact_coord for segment 2 (RX and IRS positions swapped)
                           const std::vector<bool> *active_path = nullptr); // List of active paths, vector of length [n_path_1 * n_path_2]

    // Generate diffraction ellipsoid
    // - Each ellipsoid consists of 'n_path' diffraction paths
    // - Diffraction paths can be free, blocked or partially blocked (not calculated here)
    // - Each diffraction path has 'n_seg' segments
    // - All diffraction paths originate at 'orig' and arrive at 'dest' (these points are not duplicated in the output)
    // - Points 'orig' and 'dest' lay on the semi-major axis of the ellipsoid
    // - Generated rays sample the volume of the ellipsoid
    // - Weights are calculated from the Knife-edge diffraction model when parts of the ellipsoid are shadowed
    // - Initial weights are normalized such that their sum is 1.0
    // - There are 4 levels of detail:
    //      lod = 1 : n_path =  7, n_seg = 3
    //      lod = 2 : n_path = 19, n_seg = 3
    //      lod = 3 : n_path = 37, n_seg = 4
    //      lod = 4 : n_path = 61, n_seg = 5
    //      lod = 5 : n_path =  1, n_seg = 2 (for debugging)
    //      lod = 6 : n_path =  2, n_seg = 2 (for debugging)
    // - Optional estimation of the diffracted path interaction coordinates by supplying a gain (0-1) for each path (0=blocked, 1=free)
    template <typename dtype>                                     // float or double
    void generate_diffraction_paths(const arma::Mat<dtype> *orig, // Origin points of the ellipsoid, Size [ n_pos, 3 ]
                                    const arma::Mat<dtype> *dest, // Destination points of the ellipsoid [ n_pos, 3 ]
                                    dtype center_frequency,       // Frequency in [Hz]
                                    int lod,                      // Level of detail: Scalar 1-6
                                    arma::Cube<dtype> *ray_x,     // X-Coordinate of the generated rays, Size [ n_pos, n_path, n_seg-1 ]
                                    arma::Cube<dtype> *ray_y,     // Y-Coordinate of the generated rays, Size [ n_pos, n_path, n_seg-1 ]
                                    arma::Cube<dtype> *ray_z,     // Z-Coordinate of the generated rays, Size [ n_pos, n_path, n_seg-1 ]
                                    arma::Cube<dtype> *weight);   // Weights, Size [ n_pos, n_path, n_seg ]

    // Construct a geodesic polyhedron (icosphere), a convex polyhedron made from triangles
    // - Returns the number of faces
    // - The optional output "direction" can have 2 Formats: Spherical or Cartesian
    // - For spherical directions, the values of "direction" are in the order [ v1az, v1el, v2az, v2el, v3az, v3el ]
    // - For Cartesian directions, the order is [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]
    template <typename dtype>                                    // Allowed types: float or double
    arma::uword icosphere(arma::uword n_div,                     // Number of sub-segments per edge, results in n_faces = 20 * n_div^2 elements
                          dtype radius,                          // Radius of the icosphere in meters
                          arma::Mat<dtype> *center,              // Pointing vector from the origin to the center of the triangle, matrix of size [no_faces, 3]
                          arma::Col<dtype> *length = nullptr,    // Length of the pointing vector "center" (slightly smaller than 1), vector of length [no_faces]
                          arma::Mat<dtype> *vert = nullptr,      // Vectors pointing from "center" to the vertices of the triangle, matrix of size [no_ray, 9], [x1 y1 z1 x2 y2 z3 x3 y3 z3]
                          arma::Mat<dtype> *direction = nullptr, // Directions of the vertex-rays; matrix of size [no_ray, 6] or [no_ray, 9]
                          bool direction_xyz = false);           // Direction format indicator: true = Cartesian, false = Spherical

    // Write a 3D model to a Mitsuba 3 XML file
    // - Mitsuba 3 is a research-oriented, retargetable rendering system: https://www.mitsuba-renderer.org
    // - NVIDIA Sionna RT is an open-source, hardware-accelerated differentiable ray tracer for radio propagation
    //   modeling, built on top of Mitsuba 3: https://developer.nvidia.com/sionna
    // - Mitsuba 3 XML files can be used to import 3D geometry into Sionna RT.
    // - This function converts a 3D mesh from quadriga-lib into the Mitsuba XML format.
    template <typename dtype>
    void mitsuba_xml_file_write(const std::string &fn,                     // Output file name
                                const arma::Mat<dtype> &vert_list,         // Vertex list, size [n_vert, 3]
                                const arma::umat &face_ind,                // Face indices (0-based), size [n_mesh, 3]
                                const arma::uvec &obj_ind,                 // Object indices (1-based), size [n_mesh]
                                const arma::uvec &mtl_ind,                 // Material indices (1-based), size [n_mesh]
                                const std::vector<std::string> &obj_names, // Object names, length = max(obj_ind)-1
                                const std::vector<std::string> &mtl_names, // Material names, length = max(mtl_ind)-1
                                const arma::Mat<dtype> &bsdf = {},         // BSDF data, size [mtl_names.size(), 17]
                                bool map_to_itu_materials = false);        // Optional mapping to ITU default materials used by Sionna

    // Read Wavefront .obj file
    // - See: https://en.wikipedia.org/wiki/Wavefront_.obj_file
    // - 3D model must be triangularized
    // - Material properties are encoded in the material name using the "usemtl [material name]" tag
    // - Default material properties are taken from ITU-R P.2040-3, Table 3
    // - Default materials: vacuum, air, itu_concrete, itu_brick, itu_plasterboard, itu_wood, itu_glass, itu_ceiling_board,
    //                      itu_ceiling_board, itu_chipboard, itu_plywood, itu_marble, itu_metal, itu_very_dry_ground,
    //                      itu_medium_dry_ground, itu_wet_ground, itu_vegetation, itu_water, itu_ice, irr_glass
    // - Supported frequency range: 1 - 40 GHz (1 - 10 GHz for ground materials)
    // - Custom materials can be defined by: "usemtl Name::A:B:C:D:att"
    //          - Real part of relative permittivity "eta = A * fGHz ^ B"
    //          - Conductivity "sigma = C * fGHz ^ D"
    //          - att = Material penetration loss in dB (fixed loss per material interaction)
    // - Unknown materials are treated as Vacuum
    // - Returns number of mesh elements "n_mesh"
    // - Attempts to change the size of the output if it does not have correct size already
    template <typename dtype>                                                // Supported types: float or double
    arma::uword obj_file_read(std::string fn,                                // File name
                              arma::Mat<dtype> *mesh = nullptr,              // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                              arma::Mat<dtype> *mtl_prop = nullptr,          // Material properties, Size: [ n_mesh, 5 ]
                              arma::Mat<dtype> *vert_list = nullptr,         // List of vertices found in the OBJ file, Size: [ n_vert, 3 ]
                              arma::umat *face_ind = nullptr,                // Vertex indices matching the corresponding mesh elements, 0-based, Size: [ n_mesh, 3 ]
                              arma::uvec *obj_ind = nullptr,                 // Object index, 1-based, Size: [ n_mesh ]
                              arma::uvec *mtl_ind = nullptr,                 // Material index, 1-based, Size: [ n_mesh ]
                              std::vector<std::string> *obj_names = nullptr, // Object names, Size: [ max(obj_ind) - 1 ]
                              std::vector<std::string> *mtl_names = nullptr, // Material names, Size: [ max(mtl_ind) - 1 ]
                              arma::Mat<dtype> *bsdf = nullptr);             // BSDF data from .MTL File, size [mtl_names.size, 15]

    // Tests if 3D objects overlap (have a shared volume or boolean intersection)
    // - Returns: Subset of list of object indices (obj_ind) that are overlapping, length [ n_overlap ]
    template <typename dtype>                                               // Supported types: float or double
    arma::uvec obj_overlap_test(const arma::Mat<dtype> *mesh,               // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                                const arma::uvec *obj_ind,                  // Object index, 1-based, Size: [ n_mesh ]
                                std::vector<std::string> *reason = nullptr, // Optional output: Overlap reason, Length [ n_overlap ]
                                dtype tolerance = 0.0005);                  // Optional input: Detection tolerance in meters

    // Convert paths to tubes
    // - Paths are defined by a list of ordered points
    // - This function adds faced around the paths for rendereing, e.g. in Blender
    // - Faces are provided as quads
    // - Edges of the faces lie on a circle around the path with a given radius
    // - Internal computations are done in double precision for accuracy
    template <typename dtype>
    void path_to_tube(const arma::Mat<dtype> *path_coord, // Path coordinates, size [3, n_coord ]
                      arma::Mat<dtype> *vert,             // Output: Vertices of the tube, size [3, n_coord * n_edges ]
                      arma::umat *faces,                  // Output: Face indices, 0-based, size [4, (n_coord-1) * n_edges]
                      dtype radius = 1.0,                 // Tube radius in meters
                      arma::uword n_edges = 5);           // Number of points in the circle building the tube, must be >= 3

    // Calculate the axis-aligned bounding box (AABB) of a point cloud
    // - The point cloud can be composed of sub-clouds, where each new sub-cloud is indicated by an index (= starting row number in points list)
    // - Returns a [ n_sub, 6 ] matrix with rows containing [ x_min, x_max, y_min, y_max, z_min, z_max ] of each sub-cloud
    // - The number of rows n_sub is a multiple of vec_size, padded with zeros
    template <typename dtype>
    arma::Mat<dtype> point_cloud_aabb(const arma::Mat<dtype> *points,                 // Points in 3D Space, Size: [ n_points, 3 ]
                                      const arma::u32_vec *sub_cloud_index = nullptr, // Sub-cloud index, Length: [ n_sub ]
                                      arma::uword vec_size = 1);                      // Vector size for SIMD processing (e.g. 8 for AVX2)

    // Reorganize a point cloud into smaller sub-clouds for faster processing
    // - Recursively calls "point_cloud_split" until number of elements per sub-cloud is below a target size
    // - Creates the "sub_cloud_index" indicating the start index of each sub-cloud
    // - A "vec_size" can be used to align the sub-clouds to a given vector size for SIMD processing (AVX or CUDA)
    // - For vec_size > 1, unused elements in a sub-cloud are padded with points at the center of the sub-cloud AABB
    // - "forward_index" contains the map of elements in "points" to "pointsR" in 1-based notation, padded with 0s for vec_size > 1
    // - "reverse_index" contains the map of elements in "pointsR" to "points" in 0-based notation
    // - Returns number of sub-clouds "n_sub"
    template <typename dtype>
    arma::uword point_cloud_segmentation(const arma::Mat<dtype> *points,          // Points in 3D Space (input), Size: [ n_points, 3 ]
                                         arma::Mat<dtype> *pointsR,               // Reorganized points (output), Size: [ n_pointsR, 3 ]
                                         arma::u32_vec *sub_cloud_index,          // Sub-cloud index, 0-based, Length: [ n_sub ]
                                         arma::uword target_size = 1024,          // Target value for the sub-cloud size
                                         arma::uword vec_size = 1,                // Vector size for SIMD processing (e.g. 8 for AVX2)
                                         arma::u32_vec *forward_index = nullptr,  // Index mapping elements of "points" to "pointsR", 1-based, Length: [ n_pointsR ]
                                         arma::u32_vec *reverse_index = nullptr); // Index mapping elements of "pointsR" to "points", 0-based, Length: [ n_points ]

    // Split a point cloud into two sub-clouds along a given axis
    // - Returns the axis along which the split was attempted (1 = x, 2 = y, 3 = z)
    // - Updates the output arguments pointsA and pointsB (changes size and values, invalidates data pointers)
    // - If the split failed, i.e. all elements would be in one of the two outputs, the output value is negated (-1 = x, -2 = y, -3 = z)
    //   In this case, the arguments "pointsA" and "pointsB" remain unchanged
    // - The optional output split_ind indicates into which sub-cloud (A or B) each point was put
    template <typename dtype>
    int point_cloud_split(const arma::Mat<dtype> *points,       // Points in 3D Space, Size: [ n_points, 3 ]
                          arma::Mat<dtype> *pointsA,            // First half, Size: [ n_pointsA, 9 ]
                          arma::Mat<dtype> *pointsB,            // Second half, Size: [ n_pointsB, 9 ]
                          int axis = 0,                         // Axis selector: 0 = Longest, 1 = x, 2 = y, 3 = z
                          arma::Col<int> *split_ind = nullptr); // Split indicator (optional): 1 = pointsA, 2 = pointsB, 0 = Error, Length: [ n_points ]

    // Tests whether points are inside a triangle mesh using raycasting
    // - Casts 20 rays from each point in multiple directions. If any ray intersects the mesh with a negative incidence angle,
    //   the point is classified as "inside".
    // - Returns a vector of length n_points with values: 0 for "outside", 1 for "inside".
    // - If obj_ind is provided, the return vector contains the 1-based index of the object that the point is inside.
    template <typename dtype>
    arma::u32_vec point_inside_mesh(const arma::Mat<dtype> *points,         // Points in 3D space, size: [n_points, 3]
                                    const arma::Mat<dtype> *mesh,           // Triangular mesh faces, size: [n_mesh, 9]
                                    const arma::u32_vec *obj_ind = nullptr, // Optional object indices, 1-based, size: [n_mesh]
                                    dtype distance = 0.0);                  // Optional minimum distance from objects in [m]

    // Calculate the interaction of rays (beams) with a triangle mesh
    // - A ray beam is defined by its origin, destination and a triangular wavefront (trivec and tridir)
    // - The first and second interaction point of the ray (FBS, SBS) are required (computed by ray_triangle_intersect function)
    // - Number of input rays: n_ray
    // - Only returns rays that interact with the mesh, i.e. n_rayN <= n_ray
    // - Outputs {trivecN, tridirN, orig_lengthN} will be empty if inputs {trivec, tridir, orig_length} are not provided
    // - In refraction mode, paths exhibiting 'total reflection' will have zero-power, but will be included in the output
    // - The optional input "tridir" can have 2 Formats: Spherical or Cartesian
    // - For spherical directions, the values of "tridir" are in the order [ v1az, v1el, v2az, v2el, v3az, v3el ]
    // - For Cartesian directions, the order is [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]
    // - The output "tridirN" will have the same format as the input
    template <typename dtype>
    void ray_mesh_interact(int interaction_type,                          // Interaction type: (0) Reflection, (1) Transmission, (2) Refraction
                           dtype center_frequency,                        // Center frequency in [Hz]
                           const arma::Mat<dtype> *orig,                  // Ray origin points in GCS, Size [ n_ray, 3 ]
                           const arma::Mat<dtype> *dest,                  // Ray destination points in GCS, Size [ n_ray, 3 ]
                           const arma::Mat<dtype> *fbs,                   // First interaction points in GCS, Size [ n_ray, 3 ]
                           const arma::Mat<dtype> *sbs,                   // Second interaction points in GCS, Size [ n_ray, 3 ]
                           const arma::Mat<dtype> *mesh,                  // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                           const arma::Mat<dtype> *mtl_prop,              // Material properties, Size: [ n_mesh, 5 ]
                           const arma::u32_vec *fbs_ind,                  // Index of first hit mesh element, 1-based, 0 = no hit, Size [ n_ray ]
                           const arma::u32_vec *sbs_ind,                  // Index of second hit mesh element, 1-based, 0 = no hit, Size [ n_ray ]
                           const arma::Mat<dtype> *trivec = nullptr,      // Vectors pointing from the origin to the vertices of the triangular propagation tube, Size [n_ray, 9], [x1 y1 z1 x2 y2 z3 x3 y3 z3]
                           const arma::Mat<dtype> *tridir = nullptr,      // Directions of the vertex-rays; Size Spherical [n_ray, 6], Size Cartesian [n_ray, 9]
                           const arma::Col<dtype> *orig_length = nullptr, // Path length at origin point, Size [ n_ray ]
                           arma::Mat<dtype> *origN = nullptr,             // New ray origin points in GCS, Size [ n_rayN, 3 ]
                           arma::Mat<dtype> *destN = nullptr,             // New ray destination points in GCS, Size [ n_rayN, 3 ]
                           arma::Col<dtype> *gainN = nullptr,             // Average interaction gain, Size [ n_rayN ]
                           arma::Mat<dtype> *xprmatN = nullptr,           // Polarization transfer matrix, Size [n_rayN, 8], Columns: [Re(VV), Im(VV), Re(HV), Im(HV), Re(VH), Im(VH), Re(HH), Im(HH) ]
                           arma::Mat<dtype> *trivecN = nullptr,           // Vectors pointing from the new origin to the vertices of the triangular propagation tube, Size [ n_rayN, 9 ]
                           arma::Mat<dtype> *tridirN = nullptr,           // The new directions of the vertex-rays, Size [ n_rayN, 6 ]
                           arma::Col<dtype> *orig_lengthN = nullptr,      // Path length at the new origin point, Size [ n_rayN ]
                           arma::Col<dtype> *fbs_angleN = nullptr,        // Angle between incoming ray and FBS in [rad], Size [ n_rayN ]
                           arma::Col<dtype> *thicknessN = nullptr,        // Material thickness in meters calculated from the difference between FBS and SBS, Size [ n_rayN ]
                           arma::Col<dtype> *edge_lengthN = nullptr,      // Max. edge length of the ray tube triangle at the new origin, Size [ n_rayN, 3 ]
                           arma::Mat<dtype> *normal_vecN = nullptr,       // Normal vector of FBS and SBS, Size [ n_rayN, 6 ],  order [ Nx_FBS, Ny_FBS, Nz_FBS, Nx_SBS, Ny_SBS, Nz_SBS ]
                           arma::s32_vec *out_typeN = nullptr);           // Output type code, Length [ n_rayN ]

    // Calculate the intersections of ray tubes with point clouds
    // - Returns the number of hits per point and the (0-based) indices of the rays that hit each point
    // - It is strongly recommended to use "point_cloud_segmentation" to speed up computations
    // - Returns the indices of the rays that hit the points; 0-based; Length (std::vector) [ n_points ]
    // - All internal computations are done using single precision
    template <typename dtype>
    std::vector<arma::u32_vec> ray_point_intersect(const arma::Mat<dtype> *points,                 // Points in 3D Space, Size: [ n_points, 3 ]
                                                   const arma::Mat<dtype> *orig,                   // Ray origin points in GCS, Size [ n_ray, 3 ]
                                                   const arma::Mat<dtype> *trivec,                 // Vectors pointing from the origin to the vertices of the triangular propagation tube, Size [ n_ray, 9 ]
                                                   const arma::Mat<dtype> *tridir,                 // Directions of the vertex-rays; Cartesian format; Size [ n_ray, 9 ]
                                                   const arma::u32_vec *sub_cloud_index = nullptr, // Sub-cloud index, 0-based, Optional, Length: [ n_sub ]
                                                   arma::u32_vec *hit_count = nullptr);            // Hit counter; Optional Output; Length [ n_points ]

    // Calculates the intersection of rays and triangles in three dimensions
    // - Implements the Möller–Trumbore ray-triangle intersection algorithm
    // - Uses AVX2 intrinsic functions to process 8 mesh elements in parallel
    // - All internal computations are done using single precision
    // - Instead of 'orig' and 'dest', rays can be provided as a combined object 'orig' with size [ n_ray, 6 ] = {xo, yo, zo, xd, yd, zd}
    //   The input 'dest' must be a nullptr in this case. This can help to optimize memory access patterns.
    template <typename dtype>
    void ray_triangle_intersect(const arma::Mat<dtype> *orig,                  // Ray origin points in GCS, Size [ n_ray, 3 ]
                                const arma::Mat<dtype> *dest,                  // Ray destination points in GCS, Size [ n_ray, 3 ]
                                const arma::Mat<dtype> *mesh,                  // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                                arma::Mat<dtype> *fbs = nullptr,               // First interaction points in GCS, Size [ n_ray, 3 ]
                                arma::Mat<dtype> *sbs = nullptr,               // Second interaction points in GCS, Size [ n_ray, 3 ]
                                arma::u32_vec *no_interact = nullptr,          // Number of mesh between orig and dest, Size [ n_ray ]
                                arma::u32_vec *fbs_ind = nullptr,              // Index of first hit mesh element, 1-based, 0 = no hit, Size [ n_ray ]
                                arma::u32_vec *sbs_ind = nullptr,              // Index of second hit mesh element, 1-based, 0 = no hit, Size [ n_ray ]
                                const arma::u32_vec *sub_mesh_index = nullptr, // Sub-mesh index, 0-based, (optional input), Length: [ n_sub ]
                                bool transpose_inputs = false);                // Option to transpose inputs orig, dest to [ 3, n_ray ] and mesh to [ 9, n_mesh ]

    // Subdivide rays
    // - Subdivides ray beams into 4 sub beams
    // - The input "tridir" can have 2 formats: Spherical or Cartesian
    // - For spherical directions, the values of "tridir" are in the order [ v1az, v1el, v2az, v2el, v3az, v3el ]
    // - For Cartesian directions, the order is [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]
    // - The output "tridirN" will have the same format as the input
    // - If the (optional) index is given, the output will contain 'n_rayN = 4 * n_sub' elements.
    // - If the input "dest" is not given, the corresponding output "destN" will be empty
    // - Returns the number of rays in the output 'n_rayN'
    template <typename dtype>
    arma::uword subdivide_rays(const arma::Mat<dtype> *orig,           // Ray origin points in GCS, Size [ n_ray, 3 ]
                               const arma::Mat<dtype> *trivec,         // Vectors pointing from the origin to the vertices of the triangular propagation tube, Size [ n_ray, 9 ]
                               const arma::Mat<dtype> *tridir,         // Directions of the vertex-rays; matrix of size [no_ray, 6] or [no_ray, 9]
                               const arma::Mat<dtype> *dest = nullptr, // Ray destination points in GCS, Size [ n_ray, 3 ]
                               arma::Mat<dtype> *origN = nullptr,      // New ray origin points in GCS, Size [ n_rayN, 3 ]
                               arma::Mat<dtype> *trivecN = nullptr,    // Vectors pointing from the new origin to the vertices of the triangular propagation tube, Size [ n_rayN, 9 ]
                               arma::Mat<dtype> *tridirN = nullptr,    // The new directions of the vertex-rays, Size [ n_rayN, 6 ]
                               arma::Mat<dtype> *destN = nullptr,      // New ray destination points in GCS, Size [ n_rayN, 3 ]
                               const arma::u32_vec *index = nullptr,   // Optional list of indices, 0-based, Length [ n_ind ]
                               const double ray_offset = 0.0);         // Offset of new ray origin from face plane

    // Subdivide triangles into smaller triangles
    // - Returns the number of triangles after subdivision
    // - Attempts to change the size of the output if it does not match [n_triangles_out, 9]
    // - Optional processing of materials (copy of the original material)
    template <typename dtype>                                                   // Supported types: float or double
    arma::uword subdivide_triangles(arma::uword n_div,                          // Number of divisions per edge, results in: n_triangles_out = n_triangles_in * n_div^2
                                    const arma::Mat<dtype> *triangles_in,       // Input, matrix of size [n_triangles_in, 9]
                                    arma::Mat<dtype> *triangles_out,            // Output, matrix of size [n_triangles_out, 9]
                                    const arma::Mat<dtype> *mtl_prop = nullptr, // Material properties (input), Size: [ n_triangles_in, 5 ], optional
                                    arma::Mat<dtype> *mtl_prop_out = nullptr);  // Material properties (output), Size: [ n_triangles_out, 5 ], optional

    // Calculate the axis-aligned bounding box (AABB) of a 3D mesh
    // - The mesh can be composed of sub-meshes, where each new sub_mesh is indicated by an index (=row number)
    // - Output is a [ n_sub, 6 ] matrix with rows containing [ x_min, x_max, y_min, y_max, z_min, z_max ] of each sub-mesh
    template <typename dtype>
    arma::Mat<dtype> triangle_mesh_aabb(const arma::Mat<dtype> *mesh,                  // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                                        const arma::u32_vec *sub_mesh_index = nullptr, // Sub-mesh index, Length: [ n_sub ]
                                        arma::uword vec_size = 1);                     // Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA)

    // Reorganize a 3D mesh into smaller sub-meshes for faster processing
    // - Recursively calls "mesh_split" until number of elements per sub-mesh is below a target size
    // - Creates the "sub_mesh_index" indicating the start index of each sub-mesh
    // - A "vec_size" can be used to align the sub-meshes to a given vector size for SIMD processing (AVX or CUDA)
    // - For vec_size > 1, unused elements in a sub-mesh are padded with 0-size faces at the center of the sub-mesh AABB
    // - If "mtl_prop" is given as an input, "mtl_propR" contains the reorganized materials, padded with "Air" for vec_size > 1
    // - "mesh_index" contains the map of elements in "mesh" to "meshR" in 1-based notation, padded with 0s for vec_size > 1
    // - Returns number of sub-meshes "n_sub"
    template <typename dtype>
    arma::uword triangle_mesh_segmentation(const arma::Mat<dtype> *mesh,               // Faces of the triangular mesh (input), Size: [ n_mesh, 9 ]
                                           arma::Mat<dtype> *meshR,                    // Reorganized mesh (output), Size: [ n_meshR, 9 ]
                                           arma::u32_vec *sub_mesh_index,              // Sub-mesh index, 0-based, Length: [ n_sub ]
                                           arma::uword target_size = 1024,             // Target value for the sub-mesh size
                                           arma::uword vec_size = 1,                   // Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA)
                                           const arma::Mat<dtype> *mtl_prop = nullptr, // Material properties (input), Size: [ n_mesh, 5 ], optional
                                           arma::Mat<dtype> *mtl_propR = nullptr,      // Material properties (output), Size: [ n_meshR, 5 ], optional
                                           arma::u32_vec *mesh_index = nullptr);       // Index mapping elements of "mesh" to "meshR", 1-based, Length: [ n_meshR ]

    // Split a 3D mesh into two sub-meshes along a given axis
    // - Returns the axis along which the split was attempted (1 = x, 2 = y, 3 = z)
    // - If the split failed, i.e. all elements would be in one of the two outputs, the output value is negated (-1 = x, -2 = y, -3 = z)
    //   In this case, the arguments "meshA" and "meshB" remain unchanged
    template <typename dtype>
    int triangle_mesh_split(const arma::Mat<dtype> *mesh,         // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                            arma::Mat<dtype> *meshA,              // First half, Size: [ n_meshA, 9 ]
                            arma::Mat<dtype> *meshB,              // Second half, Size: [ n_meshB, 9 ]
                            int axis = 0,                         // Axis selector: 0 = Longest, 1 = x, 2 = y, 3 = z
                            arma::Col<int> *split_ind = nullptr); // Split indicator (optional): 1 = meshA, 2 = meshB, 0 = Error, Length: [ n_mesh ]
}

#endif
