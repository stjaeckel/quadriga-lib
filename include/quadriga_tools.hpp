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
#include <cstdint>

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

    // Calculate diffraction gain for multiple transmit and receive positions
    template <typename dtype>                                                                       // Supported types: float or double
    void calc_diffraction_gain(const arma::Mat<dtype> &orig,                                        // TX positions; Size: [ n_pos, 3 ]
                               const arma::Mat<dtype> &dest,                                        // RX positions; Size: [ n_pos, 3 ]
                               const arma::Mat<dtype> &mesh,                                        // Triangle vertices; Size: [ no_mesh, 9 ]
                               const arma::uvec &mtl_ind,                                           // 0-based material index, Size: [n_mesh]
                               const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop, // Material properties; Length: [n_mtl]
                               dtype center_frequency,                                              // Center frequency in [Hz]
                               int lod = 2,                                                         // Level of detail, 0-6
                               arma::Col<dtype> *gain = nullptr,                                    // Diffraction gain, linear scale; Size: [ n_pos ]
                               arma::Mat<dtype> *xprmat = nullptr,                                  // Polarization transfer matrix; Size [n_pos, 8] for EM or [n_pos, 8] for scalar
                               arma::Cube<dtype> *coord = nullptr,                                  // Diffracted path coords (excl. endpoints); Size: [ 3, n_seg-1, n_pos ]
                               int verbose = 0,                                                     // Verbosity level
                               const arma::u32_vec *sub_mesh_index = nullptr,                       // Sub-mesh index, 0-based; Length: [ no_mesh ]
                               int use_kernel = 0,                                                  // Kernel: 0=auto, 1=GENERIC, 2=AVX2, 3=CUDA
                               int gpu_id = 0,                                                      // CUDA device ID, ignored otherwise
                               bool scalar_mode = false,                                            // Scalar (acoustic) mode
                               double thin_slab_threshold = 0.0);                                   // Resolve threshold on the round-trip in-slab amplitude, 0 = always resolve, 1 = never

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

    // Make a (default) cube
    template <typename dtype>
    arma::Mat<dtype> cube(const arma::vec &scale = {1.0},              // Scale; Vector of length [1] scales all axes; length [3] scales {x,y,z} independently
                          const arma::vec &rotation = {0.0, 0.0, 0.0}, // Euler rotations in [rad], Vector of length [3]; Empty = default
                          const arma::vec &location = {0.0, 0.0, 0.0}, // Location, Vector of length [3]; Empty = default
                          const arma::uword n_div = 1);                // Number of divisions

    // Make a (default) plane
    template <typename dtype>
    arma::Mat<dtype> plane(const arma::vec &scale = {1.0},              // Scale; Vector of length [1] scales all axes; length [3] scales {x,y,z} independently
                           const arma::vec &rotation = {0.0, 0.0, 0.0}, // Euler rotations in [rad], Vector of length [3]; Empty = default
                           const arma::vec &location = {0.0, 0.0, 0.0}, // Location, Vector of length [3]; Empty = default
                           const arma::uword n_div = 1);                // Number of divisions

    // Generate diffraction ellipsoid
    template <typename dtype>                                     // float or double
    void generate_diffraction_paths(const arma::Mat<dtype> &orig, // Origin points of the ellipsoid, Size [ n_pos, 3 ]
                                    const arma::Mat<dtype> &dest, // Destination points of the ellipsoid [ n_pos, 3 ]
                                    dtype center_frequency,       // Frequency in [Hz]
                                    int lod,                      // Level of detail: Scalar 1-7
                                    arma::Cube<dtype> &ray_x,     // X-Coordinate of the generated rays, Size [ n_pos, n_path, n_seg-1 ]
                                    arma::Cube<dtype> &ray_y,     // Y-Coordinate of the generated rays, Size [ n_pos, n_path, n_seg-1 ]
                                    arma::Cube<dtype> &ray_z,     // Z-Coordinate of the generated rays, Size [ n_pos, n_path, n_seg-1 ]
                                    arma::Cube<dtype> &weight);   // Weights, Size [ n_pos, n_path, n_seg ]

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

    // Read Wavefront .obj file — see obj_file_read documentation block for details
    template <typename dtype>
    arma::uword obj_file_read(const std::string &fn_obj = "",                                          // File name, must end with .obj
                              arma::Mat<dtype> *mesh = nullptr,                                        // Triangle vertices, Size: [n_mesh, 9]
                              arma::Mat<dtype> *vert_list = nullptr,                                   // Vertex list, Size: [n_vert, 3]
                              arma::umat *face_ind = nullptr,                                          // 0-based vertex indices per face, Size: [n_mesh, 3]
                              arma::uvec *obj_ind = nullptr,                                           // 0-based object index, Size: [n_mesh]
                              std::vector<std::string> *obj_names = nullptr,                           // Object names, Size: [max(obj_ind)-1]
                              arma::uvec *mtl_ind = nullptr,                                           // 0-based material index from .mtl file, Size: [n_mesh]
                              std::vector<std::string> *mtl_names = nullptr,                           // Material names from .mtl file, Size: [no_mtl = max(mtl_ind)-1]
                              arma::Mat<dtype> *bsdf = nullptr,                                        // BSDF data from .mtl file, Size: [no_mtl, 17]
                              const std::string &fn_csv = "",                                          // Optional EM/acoustic material CSV file
                              arma::uvec *csv_ind = nullptr,                                           // 0-based material index from .csv file, Size: [n_mesh]
                              std::vector<std::string> *csv_names = nullptr,                           // Material names from .csv file, Size: [n_csv = max(csv_ind)-1]
                              std::unordered_map<std::string, std::vector<dtype>> *csv_prop = nullptr, // Material properties; Keys=csv col names; Length: [n_csv]
                              bool csv_strict = false);                                                // If true, throw if material in obj is not also in csv; false defaults to first scv materia

    // Write Wavefront .obj file
    template <typename dtype>                                                                          // Supported types: float or double
    void obj_file_write(const std::string &fn = "",                                                    // File name
                        const arma::Mat<dtype> *mesh = nullptr,                                        // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                        const arma::uvec *obj_ind = nullptr,                                           // Object index, 1-based, Size: [ n_mesh ]
                        const arma::uvec *mtl_ind = nullptr,                                           // Material index, 1-based, Size: [ n_mesh ]
                        const std::vector<std::string> *obj_names = nullptr,                           // Object names, Size: [ max(obj_ind) ]
                        const std::vector<std::string> *mtl_names = nullptr,                           // Material names, Size: [ max(mtl_ind) ]
                        arma::Mat<dtype> *vert_list_out = nullptr,                                     // Out: List of vertices generated from mesh, Size: [ n_vert, 3 ]
                        arma::umat *face_ind_out = nullptr,                                            // Out: faces indices generated from mesh, 0-based, Size: [ n_mesh, 3 ]
                        const arma::Mat<dtype> *vert_list = nullptr,                                   // List of vertices found in the OBJ file, Size: [ n_vert, 3 ]
                        const arma::umat *face_ind = nullptr,                                          // Vertex indices matching the corresponding mesh elements, 0-based, Size: [ n_mesh, 3 ]
                        const arma::Mat<dtype> *bsdf = nullptr,                                        // BSDF data for the .MTL File, size [mtl_names.size, 17]
                        const dtype threshold = 0.001,                                                 // Co-location threshold for vertices, Default: 1 mm
                        const arma::uvec *csv_ind = nullptr,                                           // 1-based EM/acoustic-material index per face (0 = no material)
                        const std::vector<std::string> *csv_names = nullptr,                           // EM/acoustic material names
                        const std::unordered_map<std::string, std::vector<dtype>> *csv_prop = nullptr, // Material properties keyed by column name
                        bool csv_write_defaults = false,                                               // If `true`, also write canonical columns absent from `csv_prop`
                        bool split_loose_parts = false);                                               // If `true`, split each object into connected components

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
    // - 0 = outside, 1 = inside any object (no obj_ind), or 1-based object index (with obj_ind)
    template <typename dtype>
    arma::uvec point_inside_mesh(const arma::Mat<dtype> *points,      // Points in 3D space, size: [n_points, 3]
                                 const arma::Mat<dtype> *mesh,        // Triangular mesh faces, size: [n_mesh, 9]
                                 const arma::uvec *obj_ind = nullptr, // Optional object indices, 0-based, size: [n_mesh]
                                 dtype distance = 0.0);               // Optional minimum distance from objects in [m]

    // Calculate the interaction of rays (beams) with a triangle mesh
    template <typename dtype>
    void ray_mesh_interact(int interaction_type,                                                // 0 = EM reflect, 1 = EM transmit 2 = EM refract 3 = scalar reflect, 4 = scalar transmit, 5 = scalar refract
                           dtype center_frequency,                                              // Center frequency in [Hz]
                           const arma::Mat<dtype> *orig,                                        // Ray origins in GCS, [n_ray, 3]
                           const arma::Mat<dtype> *dest,                                        // Ray destinations in GCS, [n_ray, 3]
                           const arma::Mat<dtype> *mesh,                                        // Triangle mesh faces, [n_mesh, 9]
                           const arma::uvec *mtl_ind,                                           // 1-based material index, Size: [n_mesh]
                           const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop, // Material properties; Length: [n_mtl]
                           const arma::u32_vec *fbs_ind,                                        // 1-based FBS mesh index (0 = no hit), [n_ray]
                           const arma::u32_vec *sbs_ind,                                        // 1-based SBS mesh index (0 = no hit), [n_ray]
                           const arma::Mat<dtype> *trivec = nullptr,                            // Beam wavefront vertices relative to origin, [n_ray, 9]
                           const arma::Mat<dtype> *tridir = nullptr,                            // Vertex-ray directions, spherical [n_ray, 6] or Cartesian [n_ray, 9]
                           arma::Mat<dtype> *origN = nullptr,                                   // New origins after interaction, [n_rayN, 3]
                           arma::Mat<dtype> *destN = nullptr,                                   // New destinations after interaction, [n_rayN, 3]
                           arma::Mat<dtype> *fbsN = nullptr,                                    // First interaction points in GCS, [n_rayN, 3]
                           arma::Mat<dtype> *sbsN = nullptr,                                    // Second interaction points in GCS, [n_rayN, 3]
                           arma::Col<dtype> *gainN = nullptr,                                   // Interaction gain (linear, excl. FSPL), [n_rayN]
                           arma::Mat<dtype> *xprmatN = nullptr,                                 // Polarization transfer matrix [ReVV ImVV ReVH ImVH ReHV ImHV ReHH ImHH], [n_rayN, 8]
                           arma::Mat<dtype> *trivecN = nullptr,                                 // Updated beam wavefront vertices, [n_rayN, 9]
                           arma::Mat<dtype> *tridirN = nullptr,                                 // Updated vertex-ray directions, format matches input
                           arma::Col<dtype> *fbs_angleN = nullptr,                              // Incidence angle at FBS in [rad], [n_rayN]
                           arma::Col<dtype> *thicknessN = nullptr,                              // Material thickness (FBS-SBS distance) in [m], [n_rayN]
                           arma::Col<dtype> *edge_lengthN = nullptr,                            // Max beam triangle edge length at new origin, [n_rayN, 3]
                           arma::Mat<dtype> *normal_vecN = nullptr,                             // FBS/SBS normals [Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S], [n_rayN, 6]
                           std::vector<uint8_t> *out_typeN = nullptr,                           // Interaction type code, [n_rayN]
                           arma::Mat<dtype> *path_dirN = nullptr,                               // Refraction-correct path direction, [n_rayN, 3]
                           bool compact = false,                                                // Remove non-hits from output, key on fbs_ind != 0
                           arma::u32_vec *ray_indN = nullptr);                                  // 0-based input ray index for each output ray, [n_rayN]

    // Update inside/outside ray state and correct gainN / xprmatN
    template <typename dtype>
    void ray_state_update(int interaction_type,                                                // 0 = EM reflect, 1 = EM transmit 2 = EM refract 3 = scalar reflect, 4 = scalar transmit, 5 = scalar refract
                          dtype center_frequency,                                              // Center frequency in [Hz]
                          const arma::Mat<dtype> *orig,                                        // Ray origins in GCS, [n_ray, 3]
                          const arma::Mat<dtype> *dest,                                        // Ray destinations in GCS, [n_ray, 3]
                          const arma::Mat<dtype> *fbsN,                                        // First interaction points in GCS, [n_rayN, 3]
                          const arma::Mat<dtype> *sbsN,                                        // Second interaction points in GCS, [n_rayN, 3]
                          const arma::u32_vec *no_interact,                                    // Mesh-hit count between orig and dest, [n_ray]
                          const arma::Col<dtype> *fbs_angleN,                                  // Incidence angle at FBS in [rad]; [n_rayN]
                          const arma::Mat<dtype> *normal_vecN,                                 // FBS/SBS normals [Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]; NULL disables wedge test; [n_rayN, 6]
                          const std::vector<uint8_t> *out_typeN,                               // Interaction type code from ray_mesh_interact; [n_rayN]
                          const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop, // Material properties keyed by column name; n_mtl materials
                          const arma::Col<short> *mtl_ind_fbsN,                                // 1-based FBS material index (0 = air); [n_rayN]
                          const arma::Col<short> *mtl_ind_sbsN,                                // 1-based SBS material index (0 = air); [n_rayN]
                          const arma::Col<short> *mtl_ind_prev_in = nullptr,                   // In (read-only): previous medium (0 = outside), [n_ray]
                          const arma::Col<short> *mtl_ind_current_in = nullptr,                // In (read-only): current medium (0 = outside), [n_ray]
                          const arma::Col<short> *mtl_ind_buffer_in = nullptr,                 // In (read-only): next-transition buffer (0 = empty), [n_ray]
                          const arma::Mat<dtype> *path_dir_prev = nullptr,                     // Physical ray direction entering this segment, `[n_ray, 3]`
                          const arma::Mat<dtype> *acc_dist_in = nullptr,                       // Accumulated in-layer distance, `[n_ray, 2]`
                          arma::Col<short> *mtl_ind_prev_outN = nullptr,                       // Out: previous medium (0 = outside), written at i; [n_rayN]
                          arma::Col<short> *mtl_ind_current_outN = nullptr,                    // Out: current medium (0 = outside), written at i; [n_rayN]
                          arma::Col<short> *mtl_ind_buffer_outN = nullptr,                     // Out: next-transition buffer (0 = empty), written at i; [n_rayN]
                          arma::Col<dtype> *gainN = nullptr,                                   // In/Out: interaction gain, updated in place; [n_rayN]
                          arma::Mat<dtype> *xprmatN = nullptr,                                 // In/Out: polarization transfer matrix, updated in place; [n_rayN, 8]
                          arma::Mat<dtype> *path_dirN = nullptr,                               // In/Out: Continuation direction, updated in place; [n_rayN]
                          arma::Mat<dtype> *acc_dist_outN = nullptr,                           // Out: Accumulated VBS distance leaving this call, [n_rayN, 2]
                          std::vector<uint8_t> *resolved_typeN = nullptr,                      // Out: Resolved ray type, [n_rayN]
                          const arma::u32_vec *ray_indN = nullptr,                             // rayN -> ray map; NULL = identity (ray = rayN); [n_rayN]
                          double eps = 0.15);                                                  // Airy resolve threshold in [0, 1]; 0 = always resolve

    // Calculate in-medium gain
    template <typename dtype>
    dtype medium_gain(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop, // Material properties; Length: [n_mtl]
                      arma::uword iM,                                                      // 1-based material index
                      dtype dist,                                                          // Length of the ray inside the medium
                      dtype center_frequency);                                             // Frequency in Hz

    // Calculate lumped interface transmission gain (att + coincidence)
    template <typename dtype>
    dtype interface_gain(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop, // Material properties; Length: [n_mtl]
                         arma::uword iM,                                                      // 1-based material index of the entered material
                         dtype center_frequency);                                             // Frequency in Hz

    // Real refractive index n = Re(sqrt(eta*mu)) of a medium; iM is 1-based (0 = air -> 1.0)
    template <typename dtype>
    dtype refractive_index(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop, // Material properties; Length: [n_mtl]
                           arma::uword iM,                                                      // 1-based material index of the entered material
                           dtype center_frequency);                                             // Frequency in Hz

    // Calculate the intersections of ray tubes with point clouds
    template <typename dtype>
    void ray_point_intersect(const arma::Mat<dtype> &points,                       // Points in 3D Space, [ n_points, 3 ]
                             const arma::Mat<dtype> &orig,                         // Ray origin points in GCS,[ n_ray, 3 ]
                             const arma::Mat<dtype> &trivec,                       // Vectors pointing from the origin to the vertices of the triangular propagation tube, [ n_ray, 9 ]
                             const arma::Mat<dtype> &tridir,                       // Directions of the vertex-rays; Cartesian format; [ n_ray, 9 ]
                             std::vector<unsigned> *hit_index = nullptr,           // flat list of 0-based ray indices, [n_hit]
                             arma::u32_vec *hit_offset = nullptr,                  // 0-based start of each point's block, [n_points + 1]
                             arma::u32_vec *hit_count = nullptr,                   // Hit counter; Optional Output; Length [ n_points ]
                             std::vector<arma::u32_vec> *hits_per_point = nullptr, // Number of hits per point and the (0-based) indices of the rays that hit each point; [n_points]
                             const arma::u32_vec *sub_cloud_index = nullptr,       // Sub-cloud index, 0-based, Optional, Length: [ n_sub ]
                             int use_kernel = 0,                                   // Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA
                             int gpu_id = 0);                                      // GPU device ID for CUDA kernel, ignored otherwise

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
    // - Splits each selected beam into 4 sub-beams, writes 4*n_subdiv rays to the first output rows and returns that count
    // - All non-NULL outputs need the same number of rows: too small is re-allocated (content discarded), larger is kept and only partly overwritten, wrong column count throws
    // - Rays are selected by 'index' (all rays if it is NULL), no offset is applied to the new origins (they stay in the wavefront plane)
    template <typename dtype>
    arma::uword subdivide_rays(const arma::Mat<dtype> &orig,           // Ray origins in GCS, [n_ray, 3]
                               const arma::Mat<dtype> &trivec,         // Beam wavefront vertices relative to origin, [n_ray, 9]
                               const arma::Mat<dtype> &tridir,         // Vertex-ray directions, spherical [n_ray, 6] or Cartesian [n_ray, 9]
                               const arma::Mat<dtype> *dest = nullptr, // Optional: Ray destinations in GCS, [n_ray, 3] or empty
                               arma::Mat<dtype> *origN = nullptr,      // Output: Subdivided ray origins in GCS, [n_rayN, 3]
                               arma::Mat<dtype> *trivecN = nullptr,    // Output: Subdivided wavefront vertices, [n_rayN, 9]
                               arma::Mat<dtype> *tridirN = nullptr,    // Output: Subdivided vertex-ray directions, format of 'tridir', [n_rayN, 6 or 9]
                               arma::Mat<dtype> *destN = nullptr,      // Output: Subdivided destinations, untouched if 'dest' is not given, [n_rayN, 3]
                               const arma::u32_vec *index = nullptr,   // Optional: 0-based ray indices, may repeat, sets output order, [n_subdiv]
                               bool transposed_output = false);        // If true, all putputs are transposed: [3/6/9, n_rayN]

    // Subdivide triangles into smaller triangles
    template <typename dtype>
    arma::uword subdivide_triangles(arma::uword n_div,                    // Number of divisions per edge, results in: n_triangles_out = n_triangles_in * n_div^2
                                    const arma::Mat<dtype> *triangles_in, // Input, matrix of size [n_triangles_in, 9]
                                    arma::Mat<dtype> *triangles_out,      // Output, matrix of size [n_triangles_out, 9]
                                    const arma::uvec *mtl_ind = nullptr,  // Material indices (input); [ n_triangles_in ]
                                    arma::uvec *mtl_ind_out = nullptr);   // Material indices (output); [ n_triangles_out ]

    // Calculate the axis-aligned bounding box (AABB) of a 3D mesh
    // - The mesh can be composed of sub-meshes, where each new sub_mesh is indicated by an index (=row number)
    // - Output is a [ n_sub, 6 ] matrix with rows containing [ x_min, x_max, y_min, y_max, z_min, z_max ] of each sub-mesh
    template <typename dtype>
    arma::Mat<dtype> triangle_mesh_aabb(const arma::Mat<dtype> *mesh,                  // Faces of the triangular mesh, Size: [ n_mesh, 9 ]
                                        const arma::u32_vec *sub_mesh_index = nullptr, // Sub-mesh index, Length: [ n_sub ]
                                        arma::uword vec_size = 1);                     // Vector size for SIMD processing (e.g. 8 for AVX2)

    // Reorganize a 3D mesh into smaller sub-meshes for faster processing
    template <typename dtype>
    arma::uword triangle_mesh_segmentation(const arma::Mat<dtype> *mesh,         // Faces of the triangular mesh (input), Size: [ n_mesh, 9 ]
                                           arma::Mat<dtype> *meshR,              // Reorganized mesh (output), Size: [ n_meshR, 9 ]
                                           arma::u32_vec *sub_mesh_index,        // Sub-mesh index, 0-based, Length: [ n_sub ]
                                           arma::uword target_size = 1024,       // Target value for the sub-mesh size
                                           arma::uword vec_size = 1,             // Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA)
                                           const arma::uvec *mtl_ind = nullptr,  // Material indices (input); [ n_mesh ]
                                           arma::uvec *mtl_ind_out = nullptr,    // Material indices (output); [ n_meshR ]
                                           arma::u32_vec *mesh_index = nullptr); // Index mapping elements of "mesh" to "meshR", 1-based, Length: [ n_meshR ]

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

    // Write data to PNG file
    template <typename dtype>
    void write_png(const arma::Mat<dtype> &data, // Data matrix
                   std::string fn,               // Filename of the PNG file, string
                   std::string colormap = "jet", // Colormap
                   dtype min_val = NAN,          // Minimum value, when passing NAN, minimum in data is used
                   dtype max_val = NAN,          // Maximum value, when passing NAN, maximum data is used
                   bool log_transform = false);  // Transform data to log-domain (10*log10(data))

    // Polarization transfer (Jones) matrix threading along a ray path
    template <typename dtype>
    void xpr_update(arma::Mat<dtype> &xprmat,                 // Existing XPR matrix; Updated in-place; EM mode [8, n_ray]; Scalar mode [2, n_ray]; Initialized if empty
                    const arma::Mat<dtype> *update = nullptr, // XPR Update; [8 or 2, 1] broadcasts; empty = unitary; otherwise [8 or 2, n_rayU]
                    arma::Col<dtype> *gain = nullptr,         // The per-ray gain; Length [n_rayU]
                    bool initialize = false,                  // Initialize xprmat to unity before applying the update
                    bool normalize = false,                   // Normalize result after updating (only n_rayU updated columns, ignores rest)
                    bool apply_gain = false,                  // If true, read gain from *gain and apply it to xprmat after normalization
                    const arma::uvec *ray_index = nullptr);   // Optional rayU to ray mapping; default: 1:1 (n_ray == n_rayU)

}

#endif