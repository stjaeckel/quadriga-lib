// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

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

    // ---- Channel statistics ----

    // Calculate the empirical averaged cumulative distribution function (CDF)
    // Input data matrix has samples in rows and data sets in columns.
    // Individual CDFs are computed per column. An averaged CDF is obtained by
    // quantile-space averaging. Inf and NaN values are excluded.
    template <typename dtype>
    void acdf(const arma::Mat<dtype> &data,            // Input data, Size [n_samples, n_sets]
              arma::Col<dtype> *bins = nullptr,        // Bin centers (in/out), Length [n_bins]
              arma::Mat<dtype> *cdf_per_set = nullptr, // Individual CDFs, Size [n_bins, n_sets]
              arma::Col<dtype> *cdf_avg = nullptr,     // Averaged CDF, Length [n_bins]
              arma::Col<dtype> *mu = nullptr,          // Mean 0.1-0.9 quantiles, Length [9]
              arma::Col<dtype> *sig = nullptr,         // Std of 0.1-0.9 quantiles, Length [9]
              arma::uword n_bins = 201);               // Number of auto-generated bins

    // Calculate the RMS delay spread in [s]
    // Returns: RMS delay spread, size: [ n_cir ]
    template <typename dtype>
    arma::Col<dtype> calc_delay_spread(const std::vector<arma::Col<dtype>> &delays, // Delays in [s], Vector (n_cir) of vectors of length [n_path]
                                       const std::vector<arma::Col<dtype>> &powers, // Path powers, linear scale, Vector (n_cir) of vectors of length [n_path]
                                       dtype threshold = 100.0,                     // Threshold in [dB] relative to strongest path, paths below p_max(dB)-threshold are excluded
                                       dtype granularity = 0.0,                     // Window size in seconds to group paths in delay domain
                                       arma::Col<dtype> *mean_delay = nullptr);     // Optional output: mean delay in [s].

    // Calculate azimuth and elevation angular spreads with spherical wrapping
    // The power-weighted mean direction is rotated to the equator to decouple spreads,
    // and an optional bank angle aligns the angular distribution to its principal axes.
    template <typename dtype>
    void calc_angular_spreads_sphere(const std::vector<arma::Col<dtype>> &az,        // Azimuth angles in [rad], Vector (n_cir) of vectors of length [n_path]
                                     const std::vector<arma::Col<dtype>> &el,        // Elevation angles in [rad], Vector (n_cir) of vectors of length [n_path]
                                     const std::vector<arma::Col<dtype>> &powers,    // Path powers in [W], Vector (n_cir) of vectors of length [n_path]
                                     arma::Col<dtype> *azimuth_spread = nullptr,     // RMS azimuth angular spread in [rad], Length [n_cir]
                                     arma::Col<dtype> *elevation_spread = nullptr,   // RMS elevation angular spread in [rad], Length [n_cir]
                                     arma::Mat<dtype> *orientation = nullptr,        // Mean-angle orientation [bank;tilt;heading] in [rad], Size [3, n_cir]
                                     std::vector<arma::Col<dtype>> *phi = nullptr,   // Rotated azimuth angles in [rad], Vector (n_cir) of vectors of length [n_path]
                                     std::vector<arma::Col<dtype>> *theta = nullptr, // Rotated elevation angles in [rad], Vector (n_cir) of vectors of length [n_path]
                                     bool disable_wrapping = false,                  // Disable the rotation and use raw az/el angles for spread calculation
                                     bool calc_bank_angle = true,                    // Compute optimal bank angle analytically (only for disable_wrapping = false)
                                     dtype quantize = 0.0);                          // Angular quantization step in [deg], 0 = disabled

    // Calculate the Rician K-Factor
    // - KF = ratio of signal power in the dominant line-of-sight (LOS) path to the power in the scattered (non-line-of-sight, or NLOS) paths
    // - LOS path is identified by matching the absolute path length with the distance between TX and RX dTR
    // - All paths arriving before dTR + window_size are considered LOS and their power is added
    // - Paths arriving after dTR + window_size, are considered NLOS
    template <typename dtype>
    void calc_rician_k_factor(const std::vector<arma::Col<dtype>> &powers,      // Path powers in [W], Vector (n_cir) of vectors of length [n_path]
                              const std::vector<arma::Col<dtype>> &path_length, // Absolute path length from TX to RX phase center, Vector (n_cir) of vectors of length [n_path]
                              const arma::Mat<dtype> &tx_pos,                   // Transmitter position in Cartesian coordinates. Size [3,1] (fixed TX) or [3, n_cir] (mobile TX).
                              const arma::Mat<dtype> &rx_pos,                   // Receiver position in Cartesian coordinates. Size [3,1] (fixed RX) or [3, n_cir] (mobile RX).
                              arma::Col<dtype> *kf = nullptr,                   // Rician K-factor, linear scale, Length [n_cir]
                              arma::Col<dtype> *pg = nullptr,                   // Total path gain (sum of path-powers), Length [n_cir]
                              dtype window_size = 0.01);                        // LOS window size in meters

    // Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases
    // - Uses the aggregate power ratio method: total co-pol / total cross-pol across all paths
    // - Computes XPR in both linear (V/H) and circular (LHCP/RHCP) polarization bases
    // - Circular basis obtained via Jones matrix transformation: M_circ = T * M_lin * T^-1
    // - Only applies for NLOS paths by default; LOS identified by path_length ≈ dTR
    // - All paths with path_length < dTR + window_size are excluded unless include_los is true
    // - M is column-major with interleaved Re/Im: rows = [Re(Mvv),Im(Mvv),Re(Mhv),Im(Mhv),Re(Mvh),Im(Mvh),Re(Mhh),Im(Mhh)]
    // - M may or may not be normalized (normalization cancels in the XPR ratio)
    // - pg is always computed over all paths (including LOS), regardless of include_los
    // - If cross-polarized power is zero, XPR is set to 0 (undefined)
    template <typename dtype>
    void calc_cross_polarization_ratio(const std::vector<arma::Col<dtype>> &powers,      // Path powers in [W], Vector (n_cir) of vectors of length [n_path]
                                       const std::vector<arma::Mat<dtype>> &M,           // Polarization transfer matrix, Vector (n_cir) of matrices of size [8, n_path]
                                       const std::vector<arma::Col<dtype>> &path_length, // Absolute path length from TX to RX phase center in [m], Vector (n_cir) of vectors of length [n_path]
                                       const arma::Mat<dtype> &tx_pos,                   // Transmitter position in Cartesian coordinates, Size [3, 1] (fixed TX) or [3, n_cir] (mobile TX)
                                       const arma::Mat<dtype> &rx_pos,                   // Receiver position in Cartesian coordinates, Size [3, 1] (fixed RX) or [3, n_cir] (mobile RX)
                                       arma::Mat<dtype> *xpr = nullptr,                  // Cross-polarization ratio, linear scale, Size [n_cir, 6], Cols: 0=agg. linear, 1=V-XPR, 2=H-XPR, 3=agg. circular, 4=LHCP, 5=RHCP
                                       arma::Col<dtype> *pg = nullptr,                   // Total path gain (sum of path-powers × polarimetric powers in M), Length [n_cir]
                                       bool include_los = false,                         // Include the LOS path(s) in the XPR calculation
                                       dtype window_size = 0.01);                        // LOS window size in meters, paths within dTR + window_size are excluded

    // ---- Site-specific simulation tools ----

    // Generate colormap
    // - Returns a 64 x 3 matrix of unsigned chars
    // - Supported colormaps: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer
    arma::uchar_mat colormap(std::string map, bool high_res = false);

    // Write data to PNG file
    template <typename dtype>                    // Types: float, double
    void write_png(const arma::Mat<dtype> &data, // Data matrix
                   std::string fn,               // Filename of the PNG file, string
                   std::string colormap = "jet", // Colormap
                   dtype min_val = NAN,          // Minimum value, when passing NAN, minimum in data is used
                   dtype max_val = NAN,          // Maximum value, when passing NAN, maximum data is used
                   bool log_transform = false);  // Transform data to log-domain (10*log10(data))

    // Calculate diffraction gain for multiple transmit and receive positions
    template <typename dtype>                                                 // Supported types: float or double
    void calc_diffraction_gain(const arma::Mat<dtype> *orig,                  // TX positions; Size: [ n_pos, 3 ]
                               const arma::Mat<dtype> *dest,                  // RX positions; Size: [ n_pos, 3 ]
                               const arma::Mat<dtype> *mesh,                  // Triangle vertices; Size: [ no_mesh, 9 ]
                               const arma::Mat<dtype> *mtl_prop,              // Material properties; Size: [ no_mesh, 5 ]
                               dtype center_frequency,                        // Center frequency in [Hz]
                               int lod = 2,                                   // Level of detail, 0-6
                               arma::Col<dtype> *gain = nullptr,              // Diffraction gain, linear scale; Size: [ n_pos ]
                               arma::Cube<dtype> *coord = nullptr,            // Diffracted path coords (excl. endpoints); Size: [ 3, n_seg-1, n_pos ]
                               int verbose = 0,                               // Verbosity level
                               const arma::u32_vec *sub_mesh_index = nullptr, // Sub-mesh index, 0-based; Length: [ no_mesh ]
                               int use_kernel = 0,                            // Kernel: 0=auto, 1=GENERIC, 2=AVX2, 3=CUDA
                               int gpu_id = 0,                                // CUDA device ID, ignored otherwise
                               bool scalar_mode = false);

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
                              std::vector<std::string> *obj_names = nullptr, // Object names, Size: [ max(obj_ind) ]
                              std::vector<std::string> *mtl_names = nullptr, // Material names, Size: [ max(mtl_ind) ]
                              arma::Mat<dtype> *bsdf = nullptr,              // BSDF data from .MTL File, size [mtl_names.size, 15]
                              const std::string &materials_csv = "");        // Location of the material parameter file

    // Write Wavefront .obj file
    template <typename dtype>                                                // Supported types: float or double
    void obj_file_write(const std::string &fn = "",                          // File name
                        const arma::Mat<dtype> *mesh = nullptr,              // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                        const arma::uvec *obj_ind = nullptr,                 // Object index, 1-based, Size: [ n_mesh ]
                        const arma::uvec *mtl_ind = nullptr,                 // Material index, 1-based, Size: [ n_mesh ]
                        const std::vector<std::string> *obj_names = nullptr, // Object names, Size: [ max(obj_ind) ]
                        const std::vector<std::string> *mtl_names = nullptr, // Material names, Size: [ max(mtl_ind) ]
                        arma::Mat<dtype> *vert_list_out = nullptr,           // Out: List of vertices generated from mesh, Size: [ n_vert, 3 ]
                        arma::umat *face_ind_out = nullptr,                  // Out: faces indices generated from mesh, 0-based, Size: [ n_mesh, 3 ]
                        const arma::Mat<dtype> *vert_list = nullptr,         // List of vertices found in the OBJ file, Size: [ n_vert, 3 ]
                        const arma::umat *face_ind = nullptr,                // Vertex indices matching the corresponding mesh elements, 0-based, Size: [ n_mesh, 3 ]
                        const arma::Mat<dtype> *bsdf = nullptr,              // BSDF data for the .MTL File, size [mtl_names.size, 17]
                        const dtype threshold = 0.001);                      // Co-location threshold for vertices, Default: 1 mm

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
    template <typename dtype>
    void ray_mesh_interact(int interaction_type,                          // 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission
                           dtype center_frequency,                        // Center frequency in [Hz]
                           const arma::Mat<dtype> *orig,                  // Ray origins in GCS, [n_ray, 3]
                           const arma::Mat<dtype> *dest,                  // Ray destinations in GCS, [n_ray, 3]
                           const arma::Mat<dtype> *fbs,                   // First interaction points in GCS, [n_ray, 3]
                           const arma::Mat<dtype> *sbs,                   // Second interaction points in GCS, [n_ray, 3]
                           const arma::Mat<dtype> *mesh,                  // Triangle mesh faces, [n_mesh, 9]
                           const arma::Mat<dtype> *mtl_prop,              // Material properties, [n_mesh, 5]
                           const arma::u32_vec *fbs_ind,                  // 1-based FBS mesh index (0 = no hit), [n_ray]
                           const arma::u32_vec *sbs_ind,                  // 1-based SBS mesh index (0 = no hit), [n_ray]
                           const arma::Mat<dtype> *trivec = nullptr,      // Beam wavefront vertices relative to origin, [n_ray, 9]
                           const arma::Mat<dtype> *tridir = nullptr,      // Vertex-ray directions, spherical [n_ray, 6] or Cartesian [n_ray, 9]
                           const arma::Col<dtype> *orig_length = nullptr, // Accumulated path length at origin, [n_ray]
                           arma::Mat<dtype> *origN = nullptr,             // New origins after interaction, [n_rayN, 3]
                           arma::Mat<dtype> *destN = nullptr,             // New destinations after interaction, [n_rayN, 3]
                           arma::Col<dtype> *gainN = nullptr,             // Interaction gain (linear, excl. FSPL), [n_rayN]
                           arma::Mat<dtype> *xprmatN = nullptr,           // Polarization transfer matrix [ReVV ImVV ReVH ImVH ReHV ImHV ReHH ImHH], [n_rayN, 8]
                           arma::Mat<dtype> *trivecN = nullptr,           // Updated beam wavefront vertices, [n_rayN, 9]
                           arma::Mat<dtype> *tridirN = nullptr,           // Updated vertex-ray directions, format matches input
                           arma::Col<dtype> *orig_lengthN = nullptr,      // Accumulated path length at new origin, [n_rayN]
                           arma::Col<dtype> *fbs_angleN = nullptr,        // Incidence angle at FBS in [rad], [n_rayN]
                           arma::Col<dtype> *thicknessN = nullptr,        // Material thickness (FBS-SBS distance) in [m], [n_rayN]
                           arma::Col<dtype> *edge_lengthN = nullptr,      // Max beam triangle edge length at new origin, [n_rayN, 3]
                           arma::Mat<dtype> *normal_vecN = nullptr,       // FBS/SBS normals [Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S], [n_rayN, 6]
                           arma::s32_vec *out_typeN = nullptr);           // Interaction type code, [n_rayN]

    // Calculate in-medium gain
    template <typename dtype>
    dtype medium_gain(const arma::Mat<dtype> &mtl_prop, // Material properties, [n_mesh, 9]
                      arma::uword iM,                   // Material index, 0-based
                      dtype dist,                       // Length of the ray inside the medium
                      dtype center_frequency);          // Frequency in Hz

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
                                                   arma::u32_vec *hit_count = nullptr,             // Hit counter; Optional Output; Length [ n_points ]
                                                   int use_kernel = 0,                             // Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA
                                                   int gpu_id = 0);                                // GPU device ID for CUDA kernel, ignored otherwise

    // Calculates the intersection of rays and triangles in three dimensions
    // - Implements the Möller–Trumbore ray-triangle intersection algorithm
    // - Supports three compute kernels: GENERIC (scalar), AVX2 (SIMD), and CUDA (GPU)
    // - All internal computations are done using single precision
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
                                const arma::Mat<dtype> *aabb = nullptr,        // Axis-aligned bounding boxes for the sub-meshes, Size [ n_sub, 6 ]
                                int use_kernel = 0,                            // Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA
                                int gpu_id = 0);                               // GPU device ID for CUDA kernel, ignored otherwise

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
                                        arma::uword vec_size = 1);                     // Vector size for SIMD processing (e.g. 8 for AVX2)

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