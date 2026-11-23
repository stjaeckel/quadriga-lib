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
    // An array of antenna or acoustic elements: element patterns, positions and port weights
    template <typename dtype> // float or double
    class arrayant
    {
    public:
        std::string name = "empty"; // Name of the array

        // Field patterns of the elements in polar-spheric coordinates: Size [n_elevation, n_azimuth, n_elements]
        arma::Cube<dtype> e_theta_re;           // E-theta (vertical) field, real part; [n_elevation, n_azimuth, n_elements]
        arma::Cube<dtype> e_theta_im;           // E-theta (vertical) field, imaginary part; [n_elevation, n_azimuth, n_elements]
        arma::Cube<dtype> e_phi_re;             // E-phi (horizontal) field, real part; [n_elevation, n_azimuth, n_elements]
        arma::Cube<dtype> e_phi_im;             // E-phi (horizontal) field, imaginary part; [n_elevation, n_azimuth, n_elements]
        arma::Col<dtype> azimuth_grid;          // Azimuth angles in rad, in [-pi, pi], strictly increasing; [n_azimuth]
        arma::Col<dtype> elevation_grid;        // Elevation angles in rad, in [-pi/2, pi/2], strictly increasing; [n_elevation]
        arma::Mat<dtype> element_pos;           // Element positions in local Cartesian coords; [3, n_elements] or empty
        arma::Mat<dtype> port_weights_re;       // Port weights, real part; [n_elements, n_ports]
        arma::Mat<dtype> port_weights_im;       // Port weights, imaginary part; [n_elements, n_ports]
        dtype center_freq = dtype(299792458.0); // Center frequency
        arrayant() {};                          // Default constructor

        // Copy: the target is marked as validated when the source passes the full check; move: the target takes over
        // the mark of the source (cache pointing to the target's own data)
        arrayant(const arrayant &other);
        arrayant(arrayant &&other);
        arrayant &operator=(const arrayant &other);
        arrayant &operator=(arrayant &&other);

        // Functions to determine the size of the array properties
        arma::uword n_elevation() const; // Number of elevation angles
        arma::uword n_azimuth() const;   // Number of azimuth angles
        arma::uword n_elements() const;  // Number of elements
        arma::uword n_ports() const;     // Number of ports (columns of the port weights)

        // Reset the size to zero (the arrayant object will contain no data)
        void reset();

        // Resize an arrayant object to new dimensions
        void set_size(arma::uword n_elevation, // Number of elevation angles
                      arma::uword n_azimuth,   // Number of azimuth angles
                      arma::uword n_elements,  // Number of elements
                      arma::uword n_ports);    // Number of ports (columns of the port weights)

        // Validate integrity (full check); returns an empty string if valid or an error message otherwise
        std::string is_valid() const;

    private:
        mutable std::atomic<const dtype *> check_ptr[9] = {}; // Data pointers of the last passed validity check (cache of the quick check)
        friend struct qd_arrayant_access;                     // Library-internal access to the cache
    };

    // ---- Antennas, transducers and patterns ----

    // Calculate the beamwidth and the pointing angles of the main beam of the elements in degree
    template <typename dtype>
    void calc_beamwidth(
        const arrayant<dtype> &ant,                    // Valid arrayant
        const arma::uvec &i_element = arma::uvec(),    // Optional: 0-based element indices, duplicates allowed; empty = all; [n_out]
        double threshold_dB = 3.0,                     // Threshold in dB, positive and finite; 3 dB = FWHM
        arma::Col<dtype> *beamwidth_az_deg = nullptr,  // Optional output: azimuth beamwidth in degree; [n_out]
        arma::Col<dtype> *beamwidth_el_deg = nullptr,  // Optional output: elevation beamwidth in degree; [n_out]
        arma::Col<dtype> *az_point_ang_deg = nullptr,  // Optional output: azimuth pointing angle of the main beam in degree; [n_out]
        arma::Col<dtype> *el_point_ang_deg = nullptr); // Optional output: elevation pointing angle of the main beam in degree; [n_out]

    // Vector form: the same for each entry of a multi-frequency vector
    template <typename dtype>
    void calc_beamwidth(
        const std::vector<arrayant<dtype>> &arrayant_vec, // Non-empty vector that passes is_valid
        const arma::uvec &i_element = arma::uvec(),       // Optional: 0-based element indices, duplicates allowed; empty = all; [n_out]
        double threshold_dB = 3.0,                        // Threshold in dB, positive and finite; 3 dB = FWHM
        arma::Mat<dtype> *beamwidth_az_deg = nullptr,     // Optional output: azimuth beamwidth in degree; [n_out, n_freq]
        arma::Mat<dtype> *beamwidth_el_deg = nullptr,     // Optional output: elevation beamwidth in degree; [n_out, n_freq]
        arma::Mat<dtype> *az_point_ang_deg = nullptr,     // Optional output: azimuth pointing angle in degree; [n_out, n_freq]
        arma::Mat<dtype> *el_point_ang_deg = nullptr);    // Optional output: elevation pointing angle in degree; [n_out, n_freq]

    // Calculate the directivity of the elements in dBi
    // - Returns the directivity in dBi; [n_out]
    template <typename dtype>
    arma::Col<dtype> calc_directivity(
        const arrayant<dtype> &ant,                  // Valid arrayant
        const arma::uvec &i_element = arma::uvec()); // Optional: 0-based element indices, duplicates allowed; empty = all; [n_out]

    // Vector form: the same for each entry of a multi-frequency vector
    // - Returns the directivity in dBi; [n_out, n_freq]
    template <typename dtype>
    arma::Mat<dtype> calc_directivity(
        const std::vector<arrayant<dtype>> &arrayant_vec, // Non-empty vector that passes is_valid
        const arma::uvec &i_element = arma::uvec());      // Optional: 0-based element indices, duplicates allowed; empty = all; [n_out]

    // Apply the port weights: sum the element patterns, with the phase of the element positions, into one pattern per port
    // - Returns a new arrayant with n_ports elements and identity port weights, at the origin or with center_ports at the port centers
    template <typename dtype>
    arrayant<dtype> apply_port_weights(
        const arrayant<dtype> &ant,                                      // Valid arrayant
        const arma::Col<dtype> &azimuth_grid_new = arma::Col<dtype>(),   // Output azimuth grid in rad, in [-pi, pi], strictly increasing; defaults to input grid
        const arma::Col<dtype> &elevation_grid_new = arma::Col<dtype>(), // Output elevation grid in rad, in [-pi/2, pi/2], strictly increasing; defaults to input grid
        double propagation_speed = 299792458.0,                          // Wave propagation speed in m/s (~343 for acoustics); positive, finite
        bool center_ports = false);                                      // Place each port at the power-weighted center of its elements instead of the origin

    // Vector form: one output entry per output frequency; port weights are interpolated between the bracketing entries
    template <typename dtype>
    std::vector<arrayant<dtype>> apply_port_weights(
        const std::vector<arrayant<dtype>> &arrayant_vec,                // Non-empty vector that passes is_valid
        const arma::Col<dtype> &azimuth_grid_new = arma::Col<dtype>(),   // Output azimuth grid in rad, in [-pi, pi], strictly increasing; defaults to input grid
        const arma::Col<dtype> &elevation_grid_new = arma::Col<dtype>(), // Output elevation grid in rad, in [-pi/2, pi/2], strictly increasing; defaults to input grid
        const arma::Col<dtype> &freq_new = arma::Col<dtype>(),           // Output frequency grid in Hz; defaults to input grid
        double propagation_speed = 299792458.0,                          // Wave propagation speed in m/s (~343 for acoustics); positive, finite
        bool center_ports = false);                                      // Place each port at the power-weighted center of its elements instead of the origin

    // Join the elements of two arrays into a new array (block-diagonal port weights)
    template <typename dtype>
    arrayant<dtype> join_arrays(
        const arrayant<dtype> &ant_1,  // First arrayant; name and center frequency are taken from it
        const arrayant<dtype> &ant_2); // Second arrayant; sampling grids must match ant_1

    // Vector form: joins the entries of two multi-frequency vectors pairwise
    // - Both inputs must have equal entry counts, identical angular grids, and matching center_freq values at each index.
    // - Output inherits name, azimuth/elevation grids, and center_freq from arrayant_vec1.
    template <typename dtype>
    std::vector<arrayant<dtype>> join_arrays(
        const std::vector<arrayant<dtype>> &arrayant_vec1,  // First vector that passes is_valid
        const std::vector<arrayant<dtype>> &arrayant_vec2); // Second vector; must match entry count, grids, and center frequencies of arrayant_vec1

    // Copy a single element to one or more destination slots, in place
    template <typename dtype>
    void copy_element(
        arrayant<dtype> &ant,        // Valid arrayant; updated in-place
        arma::uword i_source,        // Index of the element to copy, 0-based
        const arma::uvec &i_target); // Target indices, 0-based; the array is enlarged to fit the largest index

    // Vector form: the same copy on every entry of a multi-frequency vector
    template <typename dtype>
    void copy_element(
        std::vector<arrayant<dtype>> &arrayant_vec, // Non-empty vector of valid arrayant objects; modified in-place
        arma::uword i_source,                       // Index of the element to copy from; must be within current element count
        const arma::uvec &i_target);                // Indices of target elements; enlarges all entries if any index exceeds current count

    // Set element positions for all entries in a multi-frequency arrayant vector
    // - Updates element_pos in-place on every entry in the vector identically.
    // - If i_element is empty, all positions are replaced and element_pos must have n_elements columns.
    // - If i_element is non-empty, only those indexed columns are updated; element_pos column count must match i_element length.
    // - All entries must have the same element count; uninitialized element_pos fields are zero-initialized before update.
    template <typename dtype>
    void set_element_pos(
        std::vector<arrayant<dtype>> &arrayant_vec,  // Non-empty vector of arrayant objects; modified in-place
        const arma::Mat<dtype> &element_pos,         // New (x, y, z) positions; [3, n_update]
        const arma::uvec &i_element = arma::uvec()); // Indices of elements to update; empty = all elements are replaced

    // Export the patterns as a coloured mesh to a Wavefront OBJ file for 3D visualization
    template <typename dtype>
    void export_pattern_obj(
        const std::string &fn,                       // Output OBJ filename; must end in .obj
        const arrayant<dtype> &ant,                  // Valid arrayant
        double directivity_range_dB = 30.0,          // Dynamic range of the visualized directivity pattern in dB
        const std::string &colormap = "jet",         // Colormap for the visualization
        double object_radius = 1.0,                  // Radius of the exported object
        arma::uword icosphere_n_div = 4,             // Icosphere subdivision count; higher = finer mesh
        const arma::uvec &i_element = arma::uvec()); // Optional: 0-based element indices to export; default: all

    // Remove zero-valued entries from the pattern data, in place, reducing its size
    template <typename dtype>
    void trim_pattern(arrayant<dtype> &ant); // Valid arrayant; updated in-place

    // Rotate the patterns by Euler angles, in place; the result equals interpolate_pattern with the same orientation
    // - A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
    // - Usage 0 and 1 adjust the sampling grid of non-uniformly sampled patterns, which needs all elements
    template <typename dtype>
    void rotate_pattern(
        arrayant<dtype> &ant,                        // Valid arrayant; updated in-place
        const arma::Mat<dtype> &orientation,         // Euler angles (bank, tilt, heading) in rad, finite; [3, 1] or [1, 3]
        unsigned usage = 0,                          // 0 = pattern+polarization, 1 = pattern, 2 = polarization; 3, 4 = 0, 1 without grid adjustment
        const arma::uvec &i_element = arma::uvec()); // Element indices, 0-based, no duplicates; empty = all

    // Vector form: rotates every entry, with grid adjustment always disabled (uniform grids across frequencies)
    // - For scalar acoustic fields (pressure stored in e_theta_re only), use usage = 1 to avoid spurious polarization effects.
    template <typename dtype>
    void rotate_pattern(
        std::vector<arrayant<dtype>> &arrayant_vec,  // Non-empty vector of arrayant objects; modified in-place
        const arma::Mat<dtype> &orientation,         // Euler angles (bank, tilt, heading) in rad, finite; [3, 1] or [1, 3]
        unsigned usage = 0,                          // Usage: 0 = pattern+polarization, 1 = pattern only, 2 = polarization only
        const arma::uvec &i_element = arma::uvec()); // Element indices, 0-based, no duplicates; empty = all

    // Interpolate the polarimetric field pattern of the elements at given azimuth and elevation angles
    // - Angles [1, n_ang]: the same for all outputs (planar wave); [n_out, n_ang]: one angle per output (spherical wave)
    // - Kernel GENERIC computes in the precision of dtype, AVX2 in single precision also for double; auto selects GENERIC
    template <typename dtype>
    void interpolate_pattern(
        const arrayant<dtype> &ant,                                 // Arrayant with n_elements elements
        const arma::Mat<dtype> &azimuth,                            // Azimuth angles in GCS in rad; [1, n_ang] or [n_out, n_ang]
        const arma::Mat<dtype> &elevation,                          // Elevation angles in GCS in rad; same size as azimuth
        arma::Mat<dtype> &V_re,                                     // Output: e-theta (vertical) field, real part; [n_out, n_ang]
        arma::Mat<dtype> &V_im,                                     // Output: e-theta (vertical) field, imaginary part; [n_out, n_ang]
        arma::Mat<dtype> &H_re,                                     // Output: e-phi (horizontal) field, real part; [n_out, n_ang]
        arma::Mat<dtype> &H_im,                                     // Output: e-phi (horizontal) field, imaginary part; [n_out, n_ang]
        const arma::uvec &i_element = arma::uvec(),                 // Optional: 0-based element indices, duplicates allowed; [n_out]
        const arma::Cube<dtype> &orientation = arma::Cube<dtype>(), // Optional: Euler angles (bank, tilt, heading) in rad; [3, 1, 1], [3, n_out, 1], [3, 1, n_ang] or [3, n_out, n_ang]
        const arma::Mat<dtype> &element_pos = arma::Mat<dtype>(),   // Optional: element positions in m, replacing ant.element_pos; [3, n_out]
        arma::Mat<dtype> *dist = nullptr,                           // Optional output: distance of each element from the wavefront plane through the origin in m; [n_out, n_ang]
        arma::Mat<dtype> *azimuth_loc = nullptr,                    // Optional output: azimuth in the local antenna frame in rad; [n_out, n_ang]
        arma::Mat<dtype> *elevation_loc = nullptr,                  // Optional output: elevation in the local antenna frame in rad; [n_out, n_ang]
        arma::Mat<dtype> *gamma = nullptr,                          // Optional output: polarization rotation angle in rad; [n_out, n_ang]
        arma::Mat<std::complex<dtype>> *V = nullptr,                // Optional output: complex e-theta field, replaces V_re / V_im (then left empty); [n_out, n_ang]
        arma::Mat<std::complex<dtype>> *H = nullptr,                // Optional output: complex e-phi field, replaces H_re / H_im (then left empty); [n_out, n_ang]
        int use_kernel = 0);                                        // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Vector form: interpolates a multi-frequency vector at given angles and frequencies
    // - Blends the spatial interpolation of the two entries that bracket each target frequency (spherical interpolation
    //   of the complex values with linear fallback); target frequencies outside the entries take the nearest entry
    // - Consecutive target frequencies with the same bracketing entries reuse the spatial results
    template <typename dtype>
    void interpolate_pattern(
        const std::vector<arrayant<dtype>> &arrayant_vec,           // Multi-frequency arrayant vector; entries need not be sorted by frequency
        const arma::Mat<dtype> &azimuth,                            // Azimuth angles in rad; [1, n_ang] or [n_out, n_ang]
        const arma::Mat<dtype> &elevation,                          // Elevation angles in rad; same size as azimuth
        arma::Cube<dtype> &V_re,                                    // Output: e-theta field, real part; [n_out, n_ang, n_freq]
        arma::Cube<dtype> &V_im,                                    // Output: e-theta field, imaginary part; [n_out, n_ang, n_freq]
        arma::Cube<dtype> &H_re,                                    // Output: e-phi field, real part; [n_out, n_ang, n_freq]
        arma::Cube<dtype> &H_im,                                    // Output: e-phi field, imaginary part; [n_out, n_ang, n_freq]
        const arma::Col<dtype> &freq_new = arma::Col<dtype>(),      // Optional: target frequencies in Hz; default: the center frequencies of the entries; [n_freq]
        const arma::uvec &i_element = arma::uvec(),                 // Optional: 0-based element indices; [n_out]
        const arma::Cube<dtype> &orientation = arma::Cube<dtype>(), // Optional: Euler angles in rad, applied at all frequencies; shapes as in the single form
        const arma::Mat<dtype> &element_pos = arma::Mat<dtype>(),   // Optional: element positions in m; default: those of entry 0; [3, n_out]
        arma::Cube<std::complex<dtype>> *V = nullptr,               // Optional output: complex e-theta field, replaces V_re / V_im (then left empty); [n_out, n_ang, n_freq]
        arma::Cube<std::complex<dtype>> *H = nullptr,               // Optional output: complex e-phi field, replaces H_re / H_im (then left empty); [n_out, n_ang, n_freq]
        int use_kernel = 0);                                        // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Validate a vector of arrayant objects for multi-frequency consistency (the vector form of arrayant::is_valid)
    // - Each entry is validated individually via its is_valid member (full check).
    // - Cross-entry checks (all vs. entry 0): azimuth/elevation grid sizes and values, number of elements, element positions, port_weights_re shape, and port_weights_im presence and size.
    // - All entries use the same form for element_pos and the port weights: all empty or all given (a mix is rejected).
    // - Pattern data, center_freq, and port weight values are not compared (expected to vary).
    // - Stops at first error and returns a message identifying the failing entry and property.
    template <typename dtype>
    std::string is_valid(const std::vector<arrayant<dtype>> &arrayant_vec); // Non-empty vector of arrayant objects to validate

    // Read arrayant objects from a QDANT file
    // - Parses a QuaDRiGa Array Antenna Exchange Format (QDANT) XML file and returns the entries with the given IDs, in the given order.
    // - Without ids, the entries listed in the layout are returned, each once, in order of first appearance (column-major scan); all entries if the file has no layout.
    // - An ID of 0 or an ID that is not in the file throws std::invalid_argument; the file is parsed once.
    // - Counterpart to qdant_write; primary mechanism for loading frequency-dependent models where center_freq on each entry identifies the frequency.
    template <typename dtype = double>
    std::vector<arrayant<dtype>> qdant_read(
        const std::string &fn,            // Path to the QDANT file; must not be empty
        const arma::uvec &ids = {},       // 1-based IDs of the entries to read, duplicates allowed; empty reads the entries in the layout
        arma::u32_mat *layout = nullptr); // Optional output: layout matrix of the entry IDs in the file

    // Write arrayant data to a QDANT (XML) file; returns the ID written
    template <typename dtype>
    unsigned qdant_write(
        const std::string &fn,                          // Output QDANT filename; must not be empty
        const arrayant<dtype> &ant,                     // Valid arrayant
        unsigned id = 0,                                // Target ID in file; 0 appends with auto-assigned ID
        const arma::u32_mat &layout = arma::u32_mat()); // Optional: matrix of the antenna IDs in the file; must reference only IDs present in the file

    // Vector form: writes the entries to a new QDANT file
    // - Each arrayant is stored with a sequential ID (1-based) in the file
    // - A layout matrix of size [n_entries, 1] with entries 1...n_entries is created automatically
    // - The file must not exist yet
    template <typename dtype>
    void qdant_write(
        const std::string &fn,                             // Filename of the QDANT file
        const std::vector<arrayant<dtype>> &arrayant_vec); // Vector of arrayant objects to write

    // ---- Antenna array generators ----

    // Generate an antenna array of a given type, panel arrays with the TR 38.901 tuples
    // - Types: 'omni', 'xpol', 'dipole' (or 'short-dipole'), 'half-wave-dipole', 'custom', 'ula', '3GPP' (or '3gpp'), 'multibeam',
    //   '3GPP-UT' (or '3gpp-ut', handheld), '3GPP-CPE' (or '3gpp-cpe')
    // - An input a type does not use throws; the result has the given center_freq
    template <typename dtype = double>
    arrayant<dtype> generate_antenna(
        const std::string &type,                            // Antenna type
        double res = NAN,                                   // Grid resolution in degree, 0.1 to 90; NaN or <= 0 = 10 (omni, xpol) or 1
        double center_freq = 299792458.0,                   // Center frequency in Hz, positive, finite; NaN = 299792458
        const arma::vec &element = {},                      // Element parameters per type (custom, 3GPP, multibeam, 3GPP-UT, 3GPP-CPE)
        const arma::uvec &panel = {},                       // [M, N, Mg, Ng] (ula, 3GPP, multibeam)
        const arma::vec &spacing = {},                      // [d_V, d_H, d_gV, d_gH] in wavelengths (ula, 3GPP, multibeam)
        const arma::uvec &pol = {},                         // [polarization, model] (ula, 3GPP, multibeam, 3GPP-UT, 3GPP-CPE)
        const arma::uvec &ports = {},                       // [M_P, N_P] (ula, 3GPP)
        const arma::mat &steer = {},                        // [tilt, pan] in degree, one row or one per panel (ula, 3GPP)
        const arma::mat &beams = {},                        // [2 or 3, n_beams] in degree; weights (multibeam)
        const arma::vec &device_size = {},                  // [X, Y, Z] in m (3GPP-UT, 3GPP-CPE)
        const arma::uvec &locations = {},                   // TR location labels, 1-based (3GPP-UT 1 to 8, 3GPP-CPE 1 to 9)
        const arrayant<dtype> &pattern = arrayant<dtype>(), // Element pattern (ula, 3GPP, 3GPP-UT, 3GPP-CPE)
        bool separate_beams = false,                        // One port per beam (multibeam)
        bool apply_weights = false);                        // Apply the port weights, one pattern per port (ula, 3GPP, multibeam)

    // Generate a parametric loudspeaker model
    // - Returns a vector of arrayant objects, one per frequency sample
    // - Each arrayant contains the complex-valued directivity pattern at that frequency
    // - Each arrayant has a single element; multi-driver speakers are built by calling this function per driver
    //   and joining the results via join_arrays and element_pos
    // - Supported driver types: "piston" (cone/dome), "horn" (constant directivity), "omni" (subwoofer)
    // - Supported radiation types: "monopole" (4pi), "hemisphere" (2pi, sealed box), "dipole" (figure-8), "cardioid"
    // - Horn parameters that are NaN (the default) or <= 0: hor_coverage_deg and ver_coverage_deg default to 90 and 60 deg, horn_control_freq is derived from radius
    // - Frequency samples are in Hz; if empty, third-octave bands from lower_cutoff to upper_cutoff are used
    // - Angular resolution is in degrees; used to generate azimuth_grid and elevation_grid
    template <typename dtype = double>
    std::vector<arrayant<dtype>> generate_speaker(
        const std::string &driver_type = "piston",                // Driver type: "piston", "horn", "omni"
        double radius = 0.05,                                     // Effective radiating radius in [m] (piston: cone, horn: mouth)
        double lower_cutoff = 80.0,                               // Lower -3 dB frequency in [Hz]
        double upper_cutoff = 12000.0,                            // Upper -3 dB frequency in [Hz]
        double lower_rolloff_slope_dB = 12.0,                     // Low-frequency rolloff in [dB/octave]
        double upper_rolloff_slope_dB = 12.0,                     // High-frequency rolloff in [dB/octave]
        double sensitivity_dB = 85.0,                             // On-axis sensitivity in [dB SPL] at 1W/1m
        const std::string &radiation_type = "hemisphere",         // Radiation type: "monopole", "hemisphere", "dipole", "cardioid"
        double hor_coverage_deg = NAN,                            // Horizontal coverage angle in [deg], horn only, NaN or <= 0 = 90
        double ver_coverage_deg = NAN,                            // Vertical coverage angle in [deg], horn only, NaN or <= 0 = 60
        double horn_control_freq = NAN,                           // Horn pattern control frequency in [Hz], NaN or <= 0 = c / (2 pi radius)
        double baffle_width = 0.15,                               // Enclosure baffle width in [m], piston only
        double baffle_height = 0.25,                              // Enclosure baffle height in [m], piston only
        const arma::Col<dtype> &center_freq = arma::Col<dtype>(), // Frequency sample points in [Hz]; empty = auto third-octave
        double angular_resolution_deg = 5.0);                     // Angular grid resolution in [deg]

}

#endif
