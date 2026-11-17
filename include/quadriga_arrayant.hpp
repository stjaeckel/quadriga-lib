// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_arrayant_H
#define quadriga_arrayant_H

#include <armadillo>
#include <cmath>
#include <atomic>
#include <string>

namespace quadriga_lib
{
    // Class for storing and manipulating array antenna models
    template <typename dtype> // float or double
    class arrayant
    {
    public:
        std::string name = "empty"; // Name of the array antenna object

        // Electric field of the array antenna elements in polar-spheric coordinates: Size [n_elevation, n_azimuth, n_elements]
        arma::Cube<dtype> e_theta_re;                // E-theta (vertical) field, real part; [n_elevation, n_azimuth, n_elements]
        arma::Cube<dtype> e_theta_im;                // E-theta (vertical) field, imaginary part; [n_elevation, n_azimuth, n_elements]
        arma::Cube<dtype> e_phi_re;                  // E-phi (horizontal) field, real part; [n_elevation, n_azimuth, n_elements]
        arma::Cube<dtype> e_phi_im;                  // E-phi (horizontal) field, imaginary part; [n_elevation, n_azimuth, n_elements]
        arma::Col<dtype> azimuth_grid;               // Azimuth angles in rad, in [-pi, pi], strictly increasing; [n_azimuth]
        arma::Col<dtype> elevation_grid;             // Elevation angles in rad, in [-pi/2, pi/2], strictly increasing; [n_elevation]
        arma::Mat<dtype> element_pos;                // Element positions in local Cartesian coords; [3, n_elements] or empty
        arma::Mat<dtype> coupling_re;                // Coupling matrix, real part; [n_elements, n_ports]
        arma::Mat<dtype> coupling_im;                // Coupling matrix, imaginary part; [n_elements, n_ports]
        dtype center_freq = dtype(299792458.0);      // Center frequency
        arrayant() {};                               // Default constructor

        // Copy: the target is marked as validated when the source passes the full check; move: the target takes over
        // the mark of the source (cache pointing to the target's own data)
        arrayant(const arrayant &other);
        arrayant(arrayant &&other);
        arrayant &operator=(const arrayant &other);
        arrayant &operator=(arrayant &&other);

        // Functions to determine the size of the array antenna properties
        arma::uword n_elevation() const; // Number of elevation angles
        arma::uword n_azimuth() const;   // Number of azimuth angles
        arma::uword n_elements() const;  // Number of antenna elements
        arma::uword n_ports() const;     // Number of ports (after coupling of elements)

        // Reset the size to zero (the arrayant object will contain no data)
        void reset();

        // Resize an arrayant object to new dimensions
        void set_size(arma::uword n_elevation, // Number of elevation angles
                      arma::uword n_azimuth,   // Number of azimuth angles
                      arma::uword n_elements,  // Number of antenna elements
                      arma::uword n_ports);    // Number of ports (after coupling of elements)

        // Validate integrity (full check); returns an empty string if valid or an error message otherwise
        std::string is_valid() const;

    private:
        mutable std::atomic<const dtype *> check_ptr[9] = {}; // Data pointers of the last passed validity check (cache of the quick check)
        friend struct qd_arrayant_access;                     // Library-internal access to the cache
    };

    // ---- Array antenna functions ----

    // Calculate the beamwidth and the pointing angles of the main beam of array antenna elements in degree
    template <typename dtype>
    void arrayant_calc_beamwidth(const arrayant<dtype> &ant,                      // Valid arrayant
                                 const arma::uvec *i_element = nullptr,           // Optional: 0-based element indices, duplicates allowed; nullptr or empty = all; [n_out]
                                 dtype threshold_dB = 3.0,                        // Threshold in dB, positive and finite; 3 dB = FWHM
                                 arma::Col<dtype> *beamwidth_az_deg = nullptr,    // Optional output: azimuth beamwidth in degree; [n_out]
                                 arma::Col<dtype> *beamwidth_el_deg = nullptr,    // Optional output: elevation beamwidth in degree; [n_out]
                                 arma::Col<dtype> *az_point_ang_deg = nullptr,    // Optional output: azimuth pointing angle of the main beam in degree; [n_out]
                                 arma::Col<dtype> *el_point_ang_deg = nullptr);   // Optional output: elevation pointing angle of the main beam in degree; [n_out]

    // Calculate the directivity of array antenna elements in dBi
    template <typename dtype>
    void arrayant_calc_directivity(const arrayant<dtype> &ant,             // Valid arrayant
                                   arma::Col<dtype> &directivity_dBi,      // Output: directivity in dBi; [n_out]
                                   const arma::uvec *i_element = nullptr); // Optional: 0-based element indices, duplicates allowed; nullptr or empty = all; [n_out]

    // Combine element patterns, positions, and coupling weights into effective radiation patterns
    // Returns new arrayant with n_ports elements (= number of columns in coupling_re/im), each holding the combined effective pattern for that port
    template <typename dtype>
    arrayant<dtype> arrayant_combine_pattern(const arrayant<dtype> &ant,                          // Valid arrayant
                                             const arma::Col<dtype> *azimuth_grid_new = nullptr,   // Output azimuth grid in rad, in [-pi, pi], strictly increasing; defaults to input grid
                                             const arma::Col<dtype> *elevation_grid_new = nullptr, // Output elevation grid in rad, in [-pi/2, pi/2], strictly increasing; defaults to input grid
                                             dtype propagation_speed = dtype(299792458.0));        // Wave propagation speed in m/s (~343 for acoustics); positive, finite

    // Concatenate the elements of two arrayants into a new arrayant (block-diagonal coupling)
    template <typename dtype>
    arrayant<dtype> arrayant_concat(const arrayant<dtype> &ant_1,  // First arrayant; name and center frequency are taken from it
                                    const arrayant<dtype> &ant_2); // Second arrayant; sampling grids must match ant_1

    // Copy a single antenna element to one or more destination slots, in place
    template <typename dtype>
    void arrayant_copy_element(arrayant<dtype> &ant,        // Valid arrayant; updated in-place
                               arma::uword i_source,        // Index of the element to copy, 0-based
                               const arma::uvec &i_target); // Target indices, 0-based; the array is enlarged to fit the largest index

    // Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization
    template <typename dtype>
    void arrayant_export_obj_file(const std::string &fn,                  // Output OBJ filename; must end in .obj
                                  const arrayant<dtype> &ant,             // Valid arrayant
                                  dtype directivity_range_dB = 30.0,      // Dynamic range of the visualized directivity pattern in dB
                                  const std::string &colormap = "jet",    // Colormap for the visualization
                                  dtype object_radius = 1.0,              // Radius of the exported object
                                  arma::uword icosphere_n_div = 4,        // Icosphere subdivision count; higher = finer mesh
                                  const arma::uvec *i_element = nullptr); // Optional: 0-based element indices to export; default: all

    // Remove zero-valued entries from antenna pattern data, in place, reducing its size
    template <typename dtype>
    void arrayant_remove_zeros(arrayant<dtype> &ant); // Valid arrayant; updated in-place

    // Rotate antenna patterns by Euler angles, in place; the result equals arrayant_interpolate with the same orientation
    // - A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
    // - Usage 0 and 1 adjust the sampling grid of non-uniformly sampled antennas, which needs all elements
    template <typename dtype>
    void arrayant_rotate_pattern(arrayant<dtype> &ant,                   // Valid arrayant; updated in-place
                                 const arma::Mat<dtype> &orientation,    // Euler angles (bank, tilt, heading) in rad, finite; [3, 1] or [1, 3]
                                 unsigned usage = 0,                     // 0 = pattern+polarization, 1 = pattern, 2 = polarization; 3, 4 = 0, 1 without grid adjustment
                                 const arma::uvec *i_element = nullptr); // Element indices, 0-based, no duplicates; nullptr or empty = all

    // Write arrayant data to a QDANT (XML) file; returns the ID written
    template <typename dtype>
    unsigned qdant_write(const std::string &fn,                  // Output QDANT filename; must not be empty
                         const arrayant<dtype> &ant,             // Valid arrayant
                         unsigned id = 0,                        // Target ID in file; 0 appends with auto-assigned ID
                         const arma::u32_mat *layout = nullptr); // Optional: matrix of the antenna IDs in the file; must reference only IDs present in the file

    // ---- Multi-frequency array antenna functions ----

    // Calculate the beamwidth and the pointing angles of the main beam in degree for each entry of a multi-frequency vector
    template <typename dtype>
    void arrayant_calc_beamwidth_multi(const std::vector<arrayant<dtype>> &arrayant_vec, // Non-empty vector that passes arrayant_is_valid_multi
                                       const arma::uvec *i_element = nullptr,            // Optional: 0-based element indices, duplicates allowed; nullptr or empty = all; [n_out]
                                       dtype threshold_dB = 3.0,                         // Threshold in dB, positive and finite; 3 dB = FWHM
                                       arma::Mat<dtype> *beamwidth_az_deg = nullptr,     // Optional output: azimuth beamwidth in degree; [n_out, n_freq]
                                       arma::Mat<dtype> *beamwidth_el_deg = nullptr,     // Optional output: elevation beamwidth in degree; [n_out, n_freq]
                                       arma::Mat<dtype> *az_point_ang_deg = nullptr,     // Optional output: azimuth pointing angle in degree; [n_out, n_freq]
                                       arma::Mat<dtype> *el_point_ang_deg = nullptr);    // Optional output: elevation pointing angle in degree; [n_out, n_freq]

    // Calculate the directivity in dBi for each entry of a multi-frequency vector
    template <typename dtype>
    void arrayant_calc_directivity_multi(const std::vector<arrayant<dtype>> &arrayant_vec, // Non-empty vector that passes arrayant_is_valid_multi
                                         arma::Mat<dtype> &directivity_dBi,                // Output: directivity in dBi; [n_out, n_freq]
                                         const arma::uvec *i_element = nullptr);           // Optional: 0-based element indices, duplicates allowed; nullptr or empty = all; [n_out]

    // Combine element patterns, positions, and coupling weights into effective radiation patterns
    template <typename dtype>
    std::vector<arrayant<dtype>> arrayant_combine_pattern_multi(const std::vector<arrayant<dtype>> &arrayant_vec,     // Non-empty vector of valid arrayant objects
                                                                const arma::Col<dtype> *azimuth_grid_new = nullptr,   // Output azimuth grid in rad, in [-pi, pi], strictly increasing; defaults to input grid
                                                                const arma::Col<dtype> *elevation_grid_new = nullptr, // Output elevation grid in rad, in [-pi/2, pi/2], strictly increasing; defaults to input grid
                                                                const arma::Col<dtype> *freq_new = nullptr,           // Output frequency grid in Hz; defaults to input grid
                                                                dtype propagation_speed = dtype(299792458.0));        // Wave propagation speed in m/s (~343 for acoustics); positive, finite
    // Concatenate two multi-frequency arrayant vectors into a single multi-element model
    // - Both inputs must have equal entry counts, identical angular grids, and matching center_freq values at each index.
    // - Per frequency entry: pattern cubes are joined along the element (slice) dimension; element_pos matrices are horizontally concatenated (empty positions treated as zeros).
    // - Both inputs are validated with arrayant_is_valid_multi before processing; each output entry is validated before returning.
    // - Output inherits name, azimuth/elevation grids, and center_freq from arrayant_vec1.
    template <typename dtype>
    std::vector<arrayant<dtype>> arrayant_concat_multi(const std::vector<arrayant<dtype>> &arrayant_vec1,  // First validated, mutually consistent arrayant vector
                                                       const std::vector<arrayant<dtype>> &arrayant_vec2); // Second arrayant vector; must match entry count, grids, and center frequencies of arrayant_vec1

    // Copy an antenna element to one or more destinations across all entries in a multi-frequency arrayant vector
    // - Calls arrayant_copy_element on every entry in the vector with the same source and destination indices.
    // - If any destination index exceeds the current element count, all entries are enlarged; new elements receive an identity coupling entry.
    template <typename dtype>
    void arrayant_copy_element_multi(std::vector<arrayant<dtype>> &arrayant_vec, // Non-empty vector of valid arrayant objects; modified in-place
                                     arma::uword i_source,                       // Index of the element to copy from; must be within current element count
                                     const arma::uvec &i_target);                // Indices of target elements; enlarges all entries if any index exceeds current count

    // Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles
    // - Angles [1, n_ang]: the same for all outputs (planar wave); [n_out, n_ang]: one angle per output (spherical wave)
    // - Kernel GENERIC computes in the precision of dtype, AVX2 in single precision also for double; auto selects GENERIC
    template <typename dtype>
    void arrayant_interpolate(const arrayant<dtype> &ant,                     // Array antenna with n_elements elements
                              const arma::Mat<dtype> &azimuth,                // Azimuth angles in GCS in rad; [1, n_ang] or [n_out, n_ang]
                              const arma::Mat<dtype> &elevation,              // Elevation angles in GCS in rad; same size as azimuth
                              arma::Mat<dtype> &V_re,                         // Output: e-theta (vertical) field, real part; [n_out, n_ang]
                              arma::Mat<dtype> &V_im,                         // Output: e-theta (vertical) field, imaginary part; [n_out, n_ang]
                              arma::Mat<dtype> &H_re,                         // Output: e-phi (horizontal) field, real part; [n_out, n_ang]
                              arma::Mat<dtype> &H_im,                         // Output: e-phi (horizontal) field, imaginary part; [n_out, n_ang]
                              const arma::uvec *i_element = nullptr,          // Optional: 0-based element indices, duplicates allowed; [n_out]
                              const arma::Cube<dtype> *orientation = nullptr, // Optional: Euler angles (bank, tilt, heading) in rad; [3, 1, 1], [3, n_out, 1], [3, 1, n_ang] or [3, n_out, n_ang]
                              const arma::Mat<dtype> *element_pos = nullptr,  // Optional: element positions in m, replacing ant.element_pos; [3, n_out]
                              arma::Mat<dtype> *dist = nullptr,               // Optional output: distance of each element from the wavefront plane through the origin in m; [n_out, n_ang]
                              arma::Mat<dtype> *azimuth_loc = nullptr,        // Optional output: azimuth in the local antenna frame in rad; [n_out, n_ang]
                              arma::Mat<dtype> *elevation_loc = nullptr,      // Optional output: elevation in the local antenna frame in rad; [n_out, n_ang]
                              arma::Mat<dtype> *gamma = nullptr,              // Optional output: polarization rotation angle in rad; [n_out, n_ang]
                              arma::Mat<std::complex<dtype>> *V = nullptr,    // Optional output: complex e-theta field, replaces V_re / V_im (then left empty); [n_out, n_ang]
                              arma::Mat<std::complex<dtype>> *H = nullptr,    // Optional output: complex e-phi field, replaces H_re / H_im (then left empty); [n_out, n_ang]
                              int use_kernel = 0);                            // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Interpolate a multi-frequency array antenna at given angles and frequencies
    // - Blends the spatial interpolation of the two entries that bracket each target frequency (spherical interpolation
    //   of the complex values with linear fallback); target frequencies outside the entries take the nearest entry
    // - Consecutive target frequencies with the same bracketing entries reuse the spatial results
    template <typename dtype>
    void arrayant_interpolate_multi(const std::vector<arrayant<dtype>> &arrayant_vec, // Multi-frequency arrayant vector; entries need not be sorted by frequency
                                    const arma::Mat<dtype> &azimuth,                  // Azimuth angles in rad; [1, n_ang] or [n_out, n_ang]
                                    const arma::Mat<dtype> &elevation,                // Elevation angles in rad; same size as azimuth
                                    arma::Cube<dtype> &V_re,                          // Output: e-theta field, real part; [n_out, n_ang, n_freq]
                                    arma::Cube<dtype> &V_im,                          // Output: e-theta field, imaginary part; [n_out, n_ang, n_freq]
                                    arma::Cube<dtype> &H_re,                          // Output: e-phi field, real part; [n_out, n_ang, n_freq]
                                    arma::Cube<dtype> &H_im,                          // Output: e-phi field, imaginary part; [n_out, n_ang, n_freq]
                                    const arma::Col<dtype> *freq_new = nullptr,       // Optional: target frequencies in Hz; default: the center frequencies of the entries; [n_freq]
                                    const arma::uvec *i_element = nullptr,            // Optional: 0-based element indices; [n_out]
                                    const arma::Cube<dtype> *orientation = nullptr,   // Optional: Euler angles in rad, applied at all frequencies; shapes as in arrayant_interpolate
                                    const arma::Mat<dtype> *element_pos = nullptr,    // Optional: element positions in m; default: those of entry 0; [3, n_out]
                                    arma::Cube<std::complex<dtype>> *V = nullptr,     // Optional output: complex e-theta field, replaces V_re / V_im (then left empty); [n_out, n_ang, n_freq]
                                    arma::Cube<std::complex<dtype>> *H = nullptr,     // Optional output: complex e-phi field, replaces H_re / H_im (then left empty); [n_out, n_ang, n_freq]
                                    int use_kernel = 0);                              // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Validate a vector of arrayant objects for multi-frequency consistency
    // - Each entry is validated individually via its is_valid member (full check).
    // - Cross-entry checks (all vs. entry 0): azimuth/elevation grid sizes and values, number of elements, element positions, coupling_re shape, and coupling_im presence and size.
    // - All entries use the same form for element_pos and coupling: all empty or all given (a mix is rejected).
    // - Pattern data, center_freq, and coupling matrix values are not compared (expected to vary).
    // - Stops at first error and returns a message identifying the failing entry and property.
    template <typename dtype>
    std::string arrayant_is_valid_multi(const std::vector<arrayant<dtype>> &arrayant_vec); // Non-empty vector of arrayant objects to validate

    // Apply Euler rotations to all entries in a multi-frequency arrayant vector, in place
    // - Calls arrayant_rotate_pattern on every entry with grid adjustment always disabled (required for uniform-grid consistency across frequencies).
    // - For scalar acoustic fields (pressure stored in e_theta_re only), use usage = 1 to avoid spurious polarization effects.
    template <typename dtype>
    void arrayant_rotate_pattern_multi(std::vector<arrayant<dtype>> &arrayant_vec, // Non-empty vector of arrayant objects; modified in-place
                                       const arma::Mat<dtype> &orientation,        // Euler angles (bank, tilt, heading) in rad, finite; [3, 1] or [1, 3]
                                       unsigned usage = 0,                         // Usage: 0 = pattern+polarization, 1 = pattern only, 2 = polarization only
                                       const arma::uvec *i_element = nullptr);     // Element indices, 0-based, no duplicates; nullptr or empty = all

    // Set element positions for all entries in a multi-frequency arrayant vector
    // - Updates element_pos in-place on every entry in the vector identically.
    // - If i_element is nullptr or empty, all positions are replaced and element_pos must have n_elements columns.
    // - If i_element is non-empty, only those indexed columns are updated; element_pos column count must match i_element length.
    // - All entries must have the same element count; uninitialized element_pos fields are zero-initialized before update.
    template <typename dtype>
    void arrayant_set_element_pos_multi(std::vector<arrayant<dtype>> &arrayant_vec, // Non-empty vector of arrayant objects; modified in-place
                                        const arma::Mat<dtype> &element_pos,        // New (x, y, z) positions; [3, n_update]
                                        const arma::uvec *i_element = nullptr);     // Indices of elements to update; nullptr or empty = all elements are replaced

    // Read an arrayant object from a QDANT file
    // - Parses a QuaDRiGa Array Antenna Exchange Format (QDANT) XML file and returns the arrayant for the given ID.
    template <typename dtype>
    arrayant<dtype> qdant_read(const std::string &fn,            // Path to the QDANT file; must not be empty
                               unsigned id = 1,                  // 1-based ID of the antenna entry to read
                               arma::u32_mat *layout = nullptr); // Output pointer filled with the file's layout matrix of element IDs

    // Read all arrayant objects from a QDANT file into a vector
    // - Reads all entries from a QDANT file by probing ID 1 to obtain the layout, then reading each unique non-zero ID in order of first appearance (column-major scan).
    // - Each unique ID is read exactly once regardless of how many times it appears in the layout.
    // - Counterpart to qdant_write_multi; primary mechanism for loading frequency-dependent models where center_freq on each entry identifies the corresponding frequency.
    template <typename dtype>
    std::vector<arrayant<dtype>> qdant_read_multi(const std::string &fn,            // Filename of the QDANT file
                                                  arma::u32_mat *layout = nullptr); // Optional output: layout of entries in the file

    // Write a vector of array antenna objects to a single QDANT file
    // - Each arrayant is stored with a sequential ID (1-based) in the file
    // - A layout matrix of size [n_entries, 1] with entries 1...n_entries is created automatically
    // - Overwrites the file if it already exists
    template <typename dtype>
    void qdant_write_multi(const std::string &fn,                             // Filename of the QDANT file
                           const std::vector<arrayant<dtype>> &arrayant_vec); // Vector of arrayant objects to write

    // ---- Array antenna generators ----

    // Generate isotropic radiator with vertical polarization
    // - Optional input res is the resolutions of the antenna pattern sampling grid in degree
    // - Usage example: auto ant = quadriga_lib::generate_arrayant_omni<float>();
    template <typename dtype>
    arrayant<dtype> generate_arrayant_omni(dtype res = 1.0);

    // Generate cross-polarized isotropic radiator
    // - Optional input res is the resolutions of the antenna pattern sampling grid in degree
    template <typename dtype>
    arrayant<dtype> generate_arrayant_xpol(dtype res = 1.0);

    // Generate short dipole radiating with vertical polarization
    // - Optional input res is the resolutions of the antenna pattern sampling grid in degree
    template <typename dtype>
    arrayant<dtype> generate_arrayant_dipole(dtype res = 1.0);

    // Generate half-wave dipole radiating with vertical polarization
    // - Optional input res is the resolutions of the antenna pattern sampling grid in degree
    template <typename dtype>
    arrayant<dtype> generate_arrayant_half_wave_dipole(dtype res = 1.0);

    // Generate an antenna with a custom 3dB beam with (FWHM)
    template <typename dtype>
    arrayant<dtype> generate_arrayant_custom(dtype az_3dB = 90.0,       // Azimuth 3dB beam with in degree
                                             dtype el_3dB = 90.0,       // Elevation 3dB beam with in degree
                                             dtype rear_gain_lin = 0.0, // Front-back ration, linear value
                                             dtype res = 1.0);          // Resolution of the antenna pattern sampling grid in degree

    // Generate : Unified Linear Array
    // Custom pattern: It is possible to provide a custom pattern, having 1 or more elements.
    // Values for coupling, element positions and center frequency of the custom pattern are ignored.
    template <typename dtype>
    arrayant<dtype> generate_arrayant_ula(arma::uword N = 1,                        // Number of elements
                                          dtype center_freq = 299792458.0,          // The center frequency in [Hz]
                                          dtype spacing = 0.5,                      // Element spacing in [λ]
                                          const arrayant<dtype> *pattern = nullptr, // Optional custom per-element pattern
                                          dtype res = 1.0);                         // Resolution in degree, ignored if custom pattern is given

    // Generate : Antenna model for the 3GPP-NR channel model
    // Polarization indicator:
    //   1. K=1, vertical polarization only
    //   2. K=1, H/V polarized elements
    //   3. K=1, +/-45 degree polarized elements
    //   4. K=M, vertical polarization only
    //   5. K=M, H/V polarized elements
    //   6. K=M, +/-45 degree polarized elements
    // Custom pattern: It is possible to provide a custom pattern, having 1 or more elements.
    // Values for coupling, element positions and center frequency of the custom pattern are ignored.
    template <typename dtype>
    arrayant<dtype> generate_arrayant_3GPP(arma::uword M = 1,                        // Number of vertical elements
                                           arma::uword N = 1,                        // Number of horizontal elements
                                           dtype center_freq = 299792458.0,          // The center frequency in [Hz]
                                           unsigned pol = 1,                         // Polarization indicator
                                           dtype tilt = 0.0,                         // The electric downtilt angle in [deg] for pol = 4,5,6
                                           dtype spacing = 0.5,                      // Element spacing in [λ]
                                           arma::uword Mg = 1,                       // Number of nested panels in a column (Mg)
                                           arma::uword Ng = 1,                       // Number of nested panels in a row (Ng)
                                           dtype dgv = 0.5,                          // Panel spacing in vertical direction (dg,V) in [λ]
                                           dtype dgh = 0.5,                          // Panel spacing in horizontal direction (dg,H) in [λ]
                                           const arrayant<dtype> *pattern = nullptr, // Optional custom per-element pattern
                                           dtype res = 1.0);                         // Resolution in degree, ignored if custom pattern is given

    // Generate multi-beam antenna
    // - Generates a planar M x N array that forms mutiple beams pointing in different directions
    // - Beamforming weights are phasors only
    // - Polarization indicator: (1) Vertical polarization only, (2) H/V polarized elements, (3) +/-45 degree polarized elements
    template <typename dtype>
    arrayant<dtype> generate_arrayant_multibeam(arma::uword M = 1,               // Number of vertical elements
                                                arma::uword N = 1,               // Number of horizontal elements
                                                arma::Col<dtype> az = {0.0},     // Azimuth beam angles in degree
                                                arma::Col<dtype> el = {0.0},     // Elevation beam angles in degree
                                                arma::Col<dtype> weight = {1.0}, // Scaling factor for the beams
                                                dtype center_freq = 299792458.0, // The center frequency in [Hz]
                                                unsigned pol = 1,                // Polarization indicator
                                                dtype spacing = 0.5,             // Element spacing in [λ]
                                                dtype az_3dB = 120.0,            // Azimuth per-element 3dB beam with in degree
                                                dtype el_3dB = 120.0,            // Elevation per-element 3dB beam with in degree
                                                dtype rear_gain_lin = 0.0,       // Front-back ration, linear value
                                                dtype res = 1.0,                 // Resolution of the antenna pattern sampling grid in degree
                                                bool separate_beams = false,     // If true, create a separate beam for each angle pair (ignores weights)
                                                bool apply_weights = false);     // Switch to apply the beamforming weights

    // Generate a parametric loudspeaker model
    // - Returns a vector of arrayant objects, one per frequency sample
    // - Each arrayant contains the complex-valued directivity balloon at that frequency
    // - Each arrayant has a single element; multi-driver speakers are built by calling this function per driver
    //   and combining the results via arrayant_concat and element_pos
    // - Supported driver types: "piston" (cone/dome), "horn" (constant directivity), "omni" (subwoofer)
    // - Supported radiation types: "monopole" (4pi), "hemisphere" (2pi, sealed box), "dipole" (figure-8), "cardioid"
    // - Horn parameters that are NaN (the default) or <= 0: hor_coverage_deg and ver_coverage_deg default to 90 and 60 deg, horn_control_freq is derived from radius
    // - Frequency samples are in Hz; if nullptr or empty, third-octave bands from lower_cutoff to upper_cutoff are used
    // - Angular resolution is in degrees; used to generate azimuth_grid and elevation_grid
    template <typename dtype>
    std::vector<arrayant<dtype>> generate_speaker(const std::string &driver_type = "piston",         // Driver type: "piston", "horn", "omni"
                                                  dtype radius = 0.05,                               // Effective radiating radius in [m] (piston: cone, horn: mouth)
                                                  dtype lower_cutoff = 80.0,                         // Lower -3 dB frequency in [Hz]
                                                  dtype upper_cutoff = 12000.0,                      // Upper -3 dB frequency in [Hz]
                                                  dtype lower_rolloff_slope_dB = 12.0,               // Low-frequency rolloff in [dB/octave]
                                                  dtype upper_rolloff_slope_dB = 12.0,               // High-frequency rolloff in [dB/octave]
                                                  dtype sensitivity_dB = 85.0,                       // On-axis sensitivity in [dB SPL] at 1W/1m
                                                  const std::string &radiation_type = "hemisphere",  // Radiation type: "monopole", "hemisphere", "dipole", "cardioid"
                                                  dtype hor_coverage_deg = NAN,                      // Horizontal coverage angle in [deg], horn only, NaN or <= 0 = 90
                                                  dtype ver_coverage_deg = NAN,                      // Vertical coverage angle in [deg], horn only, NaN or <= 0 = 60
                                                  dtype horn_control_freq = NAN,                     // Horn pattern control frequency in [Hz], NaN or <= 0 = c / (2 pi radius)
                                                  dtype baffle_width = 0.15,                         // Enclosure baffle width in [m], piston only
                                                  dtype baffle_height = 0.25,                        // Enclosure baffle height in [m], piston only
                                                  const arma::Col<dtype> *center_freq = nullptr,     // Frequency sample points in [Hz]; nullptr or empty = auto third-octave
                                                  dtype angular_resolution_deg = 5.0);               // Angular grid resolution in [deg]

}

#endif
