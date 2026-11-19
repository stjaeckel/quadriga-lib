// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_channel_H
#define quadriga_channel_H

#include <armadillo>
#include <string>
#include <vector>
#include <any>
#include <cstring>
#include <cmath>
#include <complex>
#include <fstream>
#include <cstdint>

#include "quadriga_arrayant.hpp"

namespace quadriga_lib
{
    // Class for storing and managing channel data (+ metadata)
    // Note: "n_path" is different for each snapshot
    template <typename dtype> // float or double
    class channel
    {
    public:
        std::string name = "empty";                   // Name of the channel object
        arma::Col<dtype> center_freq;                 // Center frequency in [Hz], vector of length [1] or [n_snap] or []
        arma::Mat<dtype> tx_pos;                      // Transmitter positions, matrix of size [3, n_snap] or [3, 1]
        arma::Mat<dtype> rx_pos;                      // Receiver positions, matrix of size [3, n_snap] or [3, 1]
        arma::Mat<dtype> tx_orientation;              // Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
        arma::Mat<dtype> rx_orientation;              // Receiver orientation, matrix of size [3, n_snap] or [3, 1] or []
        std::vector<arma::Cube<dtype>> coeff_re;      // Channel coefficients, real part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> coeff_im;      // Channel coefficients, imaginary part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> delay;         // Path delays in seconds, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path] or [1, 1, n_path]
        std::vector<arma::Col<dtype>> path_gain;      // Path gain before antenna patterns, vector (n_snap) of vectors of length [n_path]
        std::vector<arma::Col<dtype>> path_length;    // Absolute path length from TX to RX phase center, vector (n_snap) of vectors of length [n_path]
        std::vector<arma::Mat<dtype>> polmat;         // Polarization transfer matrix, vector (n_snap) of matrices of size [8, n_path], interleaved complex
        std::vector<arma::Mat<dtype>> path_angles;    // Departure and arrival angles, vector (n_snap) of matrices of size [n_path, 4], {AOD, EOD, AOA, EOA}
        std::vector<arma::Mat<dtype>> fbs_pos;        // First-bounce scatterer positions, matrices of size [3, n_path]
        std::vector<arma::Mat<dtype>> lbs_pos;        // Last-bounce scatterer positions, matrices of size [3, n_path]
        std::vector<arma::Col<unsigned>> n_interact;  // Number interaction points of a path with the environment, 0 = LOS, vector (n_snap) of vectors of length [n_path]
        std::vector<arma::Mat<dtype>> interact_coord; // Interaction coordinates of paths with the environment, matrices of size [3, sum(n_interact)]
        std::vector<std::string> par_names;           // Names of unstructured data fields
        std::vector<std::any> par_data;               // Unstructured data of types {string, float, double, int, long int, arma::Col<dtype>, arma::Mat<dtype>, arma::Cube<dtype>}
        int initial_position = 0;                     // Index of reference position, values between 0 and n_snap-1 (mainly used internally)
        channel() {};                                 // Default constructor

        arma::uword n_snap() const; // Number of snapshots
        arma::uword n_rx() const;   // Number of receive antennas in coefficient matrix, returns 0 if there are no coefficients
        arma::uword n_tx() const;   // Number of transmit antennas in coefficient matrix, returns 0 if there are no coefficients
        arma::uvec n_path() const;  // Number of paths per snapshot as arma::uword
        bool empty() const;         // Returns true if the channel object contains no structured data

        // Validate integrity
        std::string is_valid() const; // Returns an empty string if channel object is valid or an error message otherwise

        // Add paths to an exisiting channel
        // - If a field exists in the calling channel object, it must also be provided to the method
        // - Number of transmit and receive antennas must match
        void add_paths(arma::uword i_snap,                                   // Snapshot index to which the paths should be added
                       const arma::Cube<dtype> *coeff_re_add = nullptr,      // Channel coefficients, real part, Cube of size [n_rx, n_tx, n_path_add]
                       const arma::Cube<dtype> *coeff_im_add = nullptr,      // Channel coefficients, imaginary part, Cube of size [n_rx, n_tx, n_path_add]
                       const arma::Cube<dtype> *delay_add = nullptr,         // Path delays in seconds, Cube of size [n_rx, n_tx, n_path_add] or [1, 1, n_path_add]
                       const arma::u32_vec *n_interact_add = nullptr,        // Number of interaction coordinates, vector of length [n_path_add]
                       const arma::Mat<dtype> *interact_coord_add = nullptr, // Interaction coordinates, matrix of size [3, sum(n_interact)]
                       const arma::Col<dtype> *path_gain_add = nullptr,      // Path gain before antenna patterns, vector of length [n_path_add]
                       const arma::Col<dtype> *path_length_add = nullptr,    // Absolute path length from TX to RX phase center, vector of length [n_path_add]
                       const arma::Mat<dtype> *polmat_add = nullptr,         // Polarization transfer matrix, Matrix of size [8, n_path_add], interleaved complex
                       const arma::Mat<dtype> *path_angles_add = nullptr,    // Departure and arrival angles, Matrix of size [n_path_add, 4], {AOD, EOD, AOA, EOA}
                       const arma::Mat<dtype> *fbs_pos_add = nullptr,        // First-bounce scatterer positions, Matrix of size [3, n_path_add]
                       const arma::Mat<dtype> *lbs_pos_add = nullptr);       // Last-bounce scatterer positions, Matrix of size [3, n_path_add]

        // Calculate the the effective path gain (linear scale)
        // - sum up the power of all paths and over all transmit and receive antenna pairs
        // - if coeff_re/im are not available, use path_gain/polarization instead (assume ideal XPOL antennas)
        // - throws an error if neither coeff_re/im nor polmat is available
        // - returns one value for each snapshot (vector with n_snap elements)
        arma::Col<dtype> calc_effective_path_gain() const;

        // Write propagation paths to a Wavefront OBJ file
        // - e.g. for visualization in Blender
        void write_paths_to_obj_file(
            const std::string &fn,                   // Filename of the OBJ file
            arma::uword max_paths = 0,               // Maximum number of paths to be shown, Default: all
            dtype gain_max_dB = -60.0,               // Maximum path gain in dB (only for color-coding)
            dtype gain_min_dB = -140.0,              // Minimum path gain in dB (only for color-coding and path selection)
            const std::string &colormap = "jet",     // Colormap for the visualization
            const arma::uvec &i_snap = arma::uvec(), // Snapshot indices, 0-based; empty = export all
            dtype radius_max = 0.05,                 // Maximum tube radius in meters
            dtype radius_min = 0.01,                 // Minimum tube radius in meters
            arma::uword n_edges = 5) const;          // Number of vertices in the circle building the tube, must be >= 3
    };

    // Class for storing and managing path data
    class path
    {
    private:
        uint8_t nFRQ = 1;                                                   // Bit 7: scalar flag; bits 0-6: number of frequencies (1-127)
        uint8_t nSEG = 0;                                                   // Number of segments
        uint8_t interact_type[2] = {0, 0};                                  // Type of the first 2 interactions
        float len_acc = 0.0f;                                               // Accumulated path length
        float xprmat[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}; // EM: Jones matrix for freq 0, col-major; SCALAR: Pressure coeffs for freqs 0-3
        // Jones labels <out><in> (= <rx><tx> in get_channels_*): ReHV = H out, V in = rows 2-3; ReVH = V out, H in = rows 4-5
        float *data = nullptr; // Variable-length data buffer

        // Data buffer EM:
        // ┌─────────────────────────────────────────────────────┐
        // │ Coordinates: [x,y,z] × nSEG = 12×nSEG bytes         │
        // ├─────────────────────────────────────────────────────┤
        // │ Jones: [ReVV,ImVV,ReHV,ImHV,ReVH,ImVH,ReHH,ImHH]    │
        // │        × nFREQ-1 = 32×(nFREQ-1) bytes               │
        // ├─────────────────────────────────────────────────────┤
        // | Interactions: 4 * ((nSEG+1) / 4) bytes, nSEG > 2    │
        // └─────────────────────────────────────────────────────┘

        // Data buffer SCALAR:
        // ┌─────────────────────────────────────────────────────┐
        // │ Coordinates: [x,y,z] × nSEG = 12×nSEG bytes         │
        // ├─────────────────────────────────────────────────────┤
        // │ Coeff: [Re,Im] × (nFREQ-4) = 8×(nFREQ-4) bytes      │
        // ├─────────────────────────────────────────────────────┤
        // | Interactions: 4 * ((nSEG+1) / 4) bytes, nSEG > 2    │
        // └─────────────────────────────────────────────────────┘

    public:
        unsigned i_cir = 0; // CIR index (= the CIR to which the path belongs)
        uint8_t n_ref = 0;  // Number of reflections
        uint8_t n_tra = 0;  // Number of transmissions / refractions
        uint8_t n_sub = 0;  // Number of subdivisions
        uint8_t n_sct = 0;  // Number of scattering events

        path() = default;                       // Default constructor: 0 segments, 1 frequency, EM mode
        path(path &&other) noexcept;            // Move constructor
        path(const path &other);                // Copy constructor
        path &operator=(path &&other) noexcept; // Move assignment
        path &operator=(const path &other);     // Copy assignment
        void free();                            // Manually clear data buffer
        ~path() { free(); }                     // Destructor

        size_t n_seg() const { return (size_t)nSEG; }         // Number of segments
        size_t n_freq() const { return size_t(nFRQ & 0x7F); } // Number of frequencies
        bool is_scalar() const { return (nFRQ & 0x80) != 0; } // Layout mode

        // Initialize data storage to a given layout
        void init(size_t segments = 0, size_t frequencies = 1, bool scalar = false);
        path(size_t segments, size_t frequencies = 1, bool scalar = false); // Init constructor

        // Access coordinates of a specific segment
        const float *coord(size_t seg = 0) const; // Read-only
        float *coord(size_t seg = 0);             // Read + write

        // Gets last segment coordinate by index (0=x, 1=y, 2=z)
        float operator()(unsigned x) const { return (nSEG > 0 && x < 3 && data) ? data[(size_t(nSEG) - 1) * 3 + x] : NAN; }

        // Calculate the total length of the path
        // D = destination after last seg, O = origin before first seg
        float length() const { return len_acc; }                                       // Get length
        void set_length(float val) { len_acc = val; }                                  // Set length
        float calc_length(float Dx, float Dy, float Dz) const;                         // member length + last to D distance
        float calc_length(float Dx, float Dy, float Dz, float Ox, float Oy, float Oz); // calculate full path length
        float calc_length(const float *D, const float *O = nullptr);                   // 3-element overload

        // Access the polarization transfer matrix (EM) or pressure coefficient (scalar) of frequency index i_freq
        const float *polmat(size_t i_freq = 0) const; // Read only
        float *polmat(size_t i_freq = 0);             // Read + write

        // Left-multiply ray's polarization transfer matrix; coeff_update must have length 8 (EM) or 2 (scalar)
        // + calculate gain from the coefficients; freq in Hz, freq > 0 applies the free-space loss
        float polmat_update(const float *coeff_update = nullptr, float gain_update = 1.0f, size_t i_freq = 0, float freq = 0.0f, float len = 0.0f);

        // Calculate gain from the coefficients; freq in Hz, freq > 0 applies the free-space loss
        float calc_gain(float freq = 0.0f, size_t i_freq = 0, float len = 0.0f) const;

        // Duplicate path, writing to "target"; return length
        float duplicate(path &target) const;

        // Duplicate path and append segment; n_ref and n_tra are not changed; return length
        // - Must be called before polmat_update/gain to use the correct path length in gain calculation
        float extend(path &target, float x, float y, float z, uint8_t type = 0) const;

        // Get/set the interaction type sequence
        std::vector<uint8_t> interaction_type_codes() const;
        void set_interaction_type_codes(const std::vector<uint8_t> &codes);
    };
    static_assert(sizeof(path) == 56, "path must be exactly 56 bytes");

    // Returns type ID of a std::any field
    // - Optional parameter dims = Pointer to 3-element array to store the size of each dimension
    // - For strings, dims[0] contains the length of the string
    // - Type-codes:
    //      -2  no value,                 -1  unsupported type,          9  std::string,
    //      10  float,                    11  double,                   12  unsigned long long int,
    //      13  long long int,            14  unsigned int,             15  int,
    //      20  arma::Mat<float>,         21  arma::Mat<double>,        22  arma::Mat<arma::uword>,
    //      23  arma::Mat<arma::sword>,   24  arma::Mat<unsigned>,      25  arma::Mat<int>,
    //      30  arma::Cube<float>,        31  arma::Cube<double>,       32  arma::Cube<arma::uword>,
    //      33  arma::Cube<arma::sword>,  34  arma::Cube<unsigned>,     35  arma::Cube<int>,
    //      40  arma::Col<float>,         41  arma::Col<double>,        42  arma::Col<arma::uword>,
    //      43  arma::Col<arma::sword>,   44  arma::Col<unsigned>,      45  arma::Col<int>,
    //      50  arma::Row<float>,         51  arma::Row<double>,        52  arma::Row<arma::uword>,
    //      53  arma::Row<arma::sword>,   54  arma::Row<unsigned>,      55  arma::Row<int>
    int any_type_id(const std::any *data, unsigned long long *dims = nullptr);

    // Create a new channel HDF file and set the index to given storage layout
    // - Throws std::runtime_error if the file exists
    // - Channels can be structured in a multi-dimensional array (up to 4D)
    //   For instance, the first dimension might represent the Base Station (BS), the second the
    //   User Equipment (UE), and the third the frequency.
    // - Dimensions of the storage layout must be defined when the file is initially created
    // - It is possible to reshape the layout using 'hdf5_reshape_layout', but the total number of
    //   entries must not change
    // - The total number of entries nx * ny * nz * nw must not exceed 4294967295 (2^32 - 1)
    void hdf5_create_file(const std::string &fn, // Filename of the HDF5 file
                          unsigned nx = 65536,   // Number of x-entries in the storage layout (first dimension, e.g. for BSs)
                          unsigned ny = 1,       // Number of y-entries in the storage layout (second dimension, e.g. for UEs)
                          unsigned nz = 1,       // Number of z-entries in the storage layout (second dimension, e.g. for carrier frequency)
                          unsigned nw = 1);      // Number of w-entries in the storage layout (fourth dimension)

    // Save data to HDF file (stand alone function)
    // - All data is stored in single precision
    // - Data is added to the end of the file
    // - The index is updated to include to the newly written data
    // - If file does not exist, a new file will be created with the layout (ix + 1, iy + 1, iz + 1, iw + 1); use
    //   hdf5_create_file for another layout
    // - Returns 0 if new dataset was created, 1 if dataset was overwritten or modified
    // - Overwriting replaces the slot completely: the old structured data and all unstructured datasets of the slot
    //   (also those written with hdf5_write_dset) are removed; the file does not shrink; copy the channels into a new
    //   file to reclaim the space
    // - One writer per file at a time (no file locking)
    // - Throws an error if the requested index in the file has not been reserved during hdf5_create_file
    template <typename dtype>                                        // Float or double
    int hdf5_write_channel(const std::string &fn,                    // Filename of the HDF5 file
                           const quadriga_lib::channel<dtype> &chan, // Channel object to write
                           unsigned ix = 0,                          // x-Index in the HDF file
                           unsigned iy = 0,                          // y-Index in the HDF file
                           unsigned iz = 0,                          // z-Index in the HDF file
                           unsigned iw = 0);                         // w-Index in the HDF file

    // Save several channels to HDF file, one per column of 'slots'
    // - If file does not exist, a new file will be created with the layout that just holds all slots (largest index+1 per dimension)
    // - All inputs are checked before anything is written: number of channels = number of columns, no slot twice, all
    //   slots inside the layout of an existing file, all channels valid
    // - Returns 0 (new) or 1 (overwritten) per slot
    template <typename dtype>                                                         // Float or double
    arma::s32_vec hdf5_write_channel_vec(const std::string &fn,                       // Filename of the HDF5 file
                                         const std::vector<channel<dtype>> &chan_vec, // Channel objects to write, [n_slot]
                                         const arma::u32_mat &slots);                 // Slot indices (ix, iy, iz, iw), 0-based, [4, n_slot]

    // Read the storage layout of channel data inside an HDF5 file
    // - Returns 4-element vector with storage layout in form: {nx, ny, nz, nw}
    // - If file does not exist, output will be {0,0,0,0}, no error-message
    // - Optional output: channel_id = vector containing the channel ID of each slot in the file, serialized, length nx × ny × nz × nw
    // - Channel IDs with value 0 indicate that there is no data stored at this index
    arma::u32_vec hdf5_read_layout(const std::string &fn, arma::u32_vec *channel_id = nullptr);

    // Reshapes the storage layout inside an existing HDF5 file
    // - Total number of elements in the HDF5 file is fixed when calling hdf5_create_file
    // - Reshaping is possible only when the total number does not change (at most 4294967295 entries)
    // - Throws if the file does not exist or is not a quadriga-lib channel file; one writer per file at a time
    void hdf5_reshape_layout(const std::string &fn, // Filename of the HDF5 file
                             unsigned nx,           // Number of x-entries in the storage layout (first dimension, e.g. for BSs)
                             unsigned ny = 1,       // Number of y-entries in the storage layout (second dimension, e.g. for UEs)
                             unsigned nz = 1,       // Number of z-entries in the storage layout (second dimension, e.g. for carrier frequency)
                             unsigned nw = 1);      // Number of w-entries in the storage layout (fourth dimension)

    // Read channel object from HDF5 file
    // - Returns empty channel object if channel ID does not exist
    // - All structured data is stored in single precision, but is converted to double in case if dtype = double
    // - Conversion to float or double is not done for unstructured data
    // - Throws std::runtime_error if the file is malformed or not a quadriga-lib channel file (files modified with
    //   other HDF5 tools are not supported and may be rejected)
    // - Optional 'i_snap': only these snapshots are read (0-based, any order, repeats allowed); fields with one entry per
    //   snapshot are sliced; throws std::invalid_argument for an index >= n_snap of a slot with snapshots; a slot
    //   without snapshots (only unstructured data) ignores 'i_snap'
    template <typename dtype = double>                                                  // Read as float or double
    channel<dtype> hdf5_read_channel(const std::string &fn,                    // Filename of the HDF5 file
                                     unsigned ix = 0,                          // x-Index in the HDF file
                                     unsigned iy = 0,                          // y-Index in the HDF file
                                     unsigned iz = 0,                          // z-Index in the HDF file
                                     unsigned iw = 0,                          // w-Index in the HDF file
                                     const arma::uvec &i_snap = arma::uvec()); // Snapshot indices, 0-based; empty = all

    // Read several channels from HDF5 file, one per column of 'slots', opening the file once
    // - Empty slots or slots outside the layout give empty channels; 'i_snap' as in hdf5_read_channel, for every slot
    template <typename dtype = double>                                                                   // Read as float or double
    std::vector<channel<dtype>> hdf5_read_channel_vec(const std::string &fn,                    // Filename of the HDF5 file
                                                      const arma::u32_mat &slots,               // Slot indices (ix, iy, iz, iw), 0-based, [4, n_slot]
                                                      const arma::uvec &i_snap = arma::uvec()); // Snapshot indices, 0-based; empty = all

    // Read single unstructured dataset from HDF5 file
    // - Besides structured channel data, extra datasets of various types can be added
    // - They are identified by a prefix (usually "par_") followed by a dataset name (name)
    // - Returns a std::any object, that can be further analyzed using the function quadriga_lib::any_type_id
    // - Returns empty std::any if there is no dataset at this location or under this name
    // - Scalar datasets are returned as scalars, 1D datasets as arma::Col (also with one element)
    // - Throws if the file does not exist or is not a quadriga-lib channel file
    std::any hdf5_read_dset(const std::string &fn,               // Filename of the HDF5 file
                            const std::string &name,             // Dataset name without the prefix (typically "par_")
                            unsigned ix = 0,                     // x-Index in the HDF file
                            unsigned iy = 0,                     // y-Index in the HDF file
                            unsigned iz = 0,                     // z-Index in the HDF file
                            unsigned iw = 0,                     // w-Index in the HDF file
                            const std::string &prefix = "par_"); // Option to set a different prefix

    // Read names of the unstructured datasets from the HDF file
    // - Returns the dataset names at the given location without the prefix, in creation order (empty if none)
    // - Throws if the file does not exist or is not a quadriga-lib channel file
    std::vector<std::string> hdf5_read_dset_names(const std::string &fn,               // Filename of the HDF5 file
                                                  unsigned ix = 0,                     // x-Index in the HDF file
                                                  unsigned iy = 0,                     // y-Index in the HDF file
                                                  unsigned iz = 0,                     // z-Index in the HDF file
                                                  unsigned iw = 0,                     // w-Index in the HDF file
                                                  const std::string &prefix = "par_"); // Option to set a different prefix

    // Write single unstructured data field to HDF5 file
    // - Scalar types: string, unsigned, int, long long, unsigned long long, float, double
    // - Armadillo Vectors, Matrices and Cubes with types: unsigned, int, unsigned long long, sword, float, double
    // - Vector types: Armadillo Row and Col types are mapped to 1D storage type
    // - arma::Row will be converted to arma::Col
    // - Parameter name may only contain letters and numbers and the underscore "_" and cannot be empty
    // - An additional prefix "par_" will be added to the name before writing to the HDF file
    // - Unsupported data types will cause an error
    // - The dataset is added to the slot, everything else in the slot is kept; an existing dataset of the same
    //   name causes an error (no overwrite)
    // - Creates the file with the layout (ix + 1, iy + 1, iz + 1, iw + 1) if it does not exist
    // - One writer per file at a time (no file locking)
    void hdf5_write_dset(const std::string &fn,               // Filename of the HDF5 file
                         const std::string &name,             // Dataset name without the prefix (typically "par_")
                         const std::any &data,                // Data to be written
                         unsigned ix = 0,                     // x-Index in the HDF file
                         unsigned iy = 0,                     // y-Index in the HDF file
                         unsigned iz = 0,                     // z-Index in the HDF file
                         unsigned iw = 0,                     // w-Index in the HDF file
                         const std::string &prefix = "par_"); // Option to set a different prefix

    // Compute the baseband frequency response of a MIMO channel
    // - Inputs are given in the time domain
    // - Calculates the discrete Fourier transform of the coefficients
    // - Outputs are returned in the frequency domain at the given sample positions
    // - Uses AVX2 to accelerate the computations (calculate 8 carriers in parallel)
    // - It is possible to call this function in a loop for multiple snapshots and parallelize this loop using OpenMP
    // - The phase bandwidth * pilot * delay is computed in double precision and reduced to one period; sin / cos,
    //   coefficients and sums are single precision
    template <typename dtype>
    void baseband_freq_response(
        const arma::Cube<dtype> &coeff_re,               // Channel coefficients, real part, cube of size [n_rx, n_tx, n_path]
        const arma::Cube<dtype> &coeff_im,               // Channel coefficients, imaginary part, cube of size [n_rx, n_tx, n_path]
        const arma::Cube<dtype> &delay,                  // Path delays in seconds, cube of size [n_rx, n_tx, n_path] or [1, 1, n_path]
        const arma::Col<dtype> &pilot_grid,              // Sub-carrier positions, relative to the bandwidth, 0.0 = fc, 1.0 = fc+bandwidth, Size: [ n_carriers ]
        const double bandwidth,                          // The baseband bandwidth in [Hz]
        arma::Cube<dtype> *hmat_re = nullptr,            // Output: Channel matrix (H), real part, Size [n_rx, n_tx, n_carriers]
        arma::Cube<dtype> *hmat_im = nullptr,            // Output: Channel matrix (H), imaginary part, Size [n_rx, n_tx, n_carriers]
        arma::Cube<std::complex<dtype>> *hmat = nullptr, // Output: Channel matrix (H), complex-valued, Size [n_rx, n_tx, n_carriers]
        int use_kernel = 0);                             // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients
    // - Interpolates multi-frequency coefficients from a coarse input grid to a dense output grid using SLERP
    // - Applies delay-induced phase rotation analytically at each output carrier frequency
    // - When remove_delay_phase is true (default), undoes the delay phase baked in by get_channels_multifreq
    //   before interpolation, then re-applies it at the output frequency; set to false for pure envelope inputs
    // - With remove_delay_phase, the channels must be generated with use_absolute_delays = true (the coefficients of
    //   get_channels_* carry the phase of the absolute path length)
    // - Delays are taken from delay[0] only (frequency-independent geometry); supports [n_rx, n_tx, n_path]
    //   or broadcastable [1, 1, n_path]
    template <typename dtype>
    void baseband_freq_response_multi(
        const std::vector<arma::Cube<dtype>> &coeff_re,  // Real part of coefficients, length [n_freq], each [n_rx, n_tx, n_path]
        const std::vector<arma::Cube<dtype>> &coeff_im,  // Imag part of coefficients, length [n_freq], each [n_rx, n_tx, n_path]
        const std::vector<arma::Cube<dtype>> &delay,     // Delays [s], length [n_freq], each [n_rx, n_tx, n_path] or [1, 1, n_path]
        const arma::Col<dtype> &freq,                    // Input sample frequencies [Hz], length [n_freq]
        const arma::Col<dtype> &freq_new,                // Output carrier frequencies [Hz], length [n_carrier]
        arma::Cube<dtype> *hmat_re = nullptr,            // Output: Channel matrix (H), real part, Size [n_rx, n_tx, n_carrier]
        arma::Cube<dtype> *hmat_im = nullptr,            // Output: Channel matrix (H), imaginary part, Size [n_rx, n_tx, n_carrier]
        arma::Cube<std::complex<dtype>> *hmat = nullptr, // Output: Channel matrix (H), complex-valued, Size [n_rx, n_tx, n_carrier]
        bool remove_delay_phase = true,                  // Remove baked-in delay phase before SLERP interpolation
        int use_kernel = 0);                             // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Compute the baseband frequency response of multiple MIMO channels
    // - Batch wrapper around [[baseband_freq_response]], applying it across snapshots in parallel via OpenMP
    // - Each element of the input vectors is a cube of shape [n_rx, n_tx, n_path]; delay supports broadcasting to [1, 1, n_path]
    // - Output vectors have length n_out: either n_snap (all snapshots) or length(i_snap) (subset)
    // - Internal arithmetic is single-precision
    template <typename dtype>
    void baseband_freq_response_vec(
        const std::vector<arma::Cube<dtype>> &coeff_re,               // Real part of time-domain channel coefficients, vector of n_snap cubes [n_rx, n_tx, n_path]
        const std::vector<arma::Cube<dtype>> &coeff_im,               // Imaginary part of time-domain channel coefficients, same structure as coeff_re
        const std::vector<arma::Cube<dtype>> &delay,                  // Path delays in seconds, same structure as coeff_re; each cube broadcastable to [1, 1, n_path]
        const arma::Col<dtype> &pilot_grid,                           // Normalized sub-carrier positions in range [0.0, 1.0]; [n_carriers]
        const double bandwidth,                                       // Total baseband bandwidth
        std::vector<arma::Cube<dtype>> *hmat_re = nullptr,            // Real part of frequency-domain channel matrices, vector of n_out cubes [n_rx, n_tx, n_carriers]
        std::vector<arma::Cube<dtype>> *hmat_im = nullptr,            // Imaginary part of frequency-domain channel matrices, same structure as hmat_re
        std::vector<arma::Cube<std::complex<dtype>>> *hmat = nullptr, // Complex-valued frequency-domain channel matrices, same structure as hmat_re
        const arma::uvec &i_snap = arma::uvec(),                      // Snapshot indices to process; if omitted, all n_snap snapshots are processed; [n_out]
        int use_kernel = 0);                                          // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Compute the wideband frequency response of multiple MIMO channels with frequency-dependent coefficients
    // - Batch wrapper around [[baseband_freq_response_multi]], applying it across snapshots in parallel via OpenMP
    // - Inputs are indexed [snapshot][input frequency]; freq and freq_new are shared by all snapshots
    // - Output vectors have length n_out: either n_snap (all snapshots) or length(i_snap) (subset)
    template <typename dtype>
    void baseband_freq_response_multi_vec(
        const std::vector<std::vector<arma::Cube<dtype>>> &coeff_re,  // Real part, n_snap vectors of n_freq cubes [n_rx, n_tx, n_path]
        const std::vector<std::vector<arma::Cube<dtype>>> &coeff_im,  // Imaginary part, same structure as coeff_re
        const std::vector<std::vector<arma::Cube<dtype>>> &delay,     // Delays [s], n_snap vectors of n_freq cubes; only entry 0 of each snapshot is used
        const arma::Col<dtype> &freq,                                 // Input sample frequencies [Hz], length [n_freq]
        const arma::Col<dtype> &freq_new,                             // Output carrier frequencies [Hz], length [n_carrier]
        std::vector<arma::Cube<dtype>> *hmat_re = nullptr,            // Output: real part, vector of n_out cubes [n_rx, n_tx, n_carrier]
        std::vector<arma::Cube<dtype>> *hmat_im = nullptr,            // Output: imaginary part, same structure as hmat_re
        std::vector<arma::Cube<std::complex<dtype>>> *hmat = nullptr, // Output: complex-valued, same structure as hmat_re
        bool remove_delay_phase = true,                               // Remove baked-in delay phase before SLERP interpolation
        const arma::uvec &i_snap = arma::uvec(),                      // Snapshot indices to process; if omitted, all n_snap snapshots are processed; [n_out]
        int use_kernel = 0);                                          // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Fixes the path delays to a grid of delay bins
    template <typename dtype>
    void quantize_delays(
        const std::vector<arma::Cube<dtype>> &coeff_re, // Channel coefficients, real part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
        const std::vector<arma::Cube<dtype>> &coeff_im, // Channel coefficients, imaginary part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
        const std::vector<arma::Cube<dtype>> &delay,    // Path delays in seconds, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path] or [1, 1, n_path];
                                                        //   finite, > -tap_spacing/2 (values below 0 count as 0) and < 2^31 * tap_spacing
        std::vector<arma::Cube<dtype>> &coeff_re_quant, // Output coefficients, real part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_taps]
        std::vector<arma::Cube<dtype>> &coeff_im_quant, // Output coefficients, imaginary part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_taps]
        std::vector<arma::Cube<dtype>> &delay_quant,    // Output delays in seconds, vector (n_snap) of Cubes of size [n_rx, n_tx, n_taps], or [1, 1, n_taps]
                                                        //   for delay input [1, 1, n_path] with fix_taps 1 or 2
        dtype tap_spacing = 5.0e-9,                     // Tap spacing in seconds
        arma::uword max_taps = 48,                      // 0 = unlimited
        dtype power_exponent = 1.0,                     // Interpolation exponent (0.5=wideband, 1.0=narrowband)
        int fix_taps = 0);                              // 0-3, delay sharing mode

    // Pre-parsed QRT file metadata cache for repeated qrt_file_read calls
    // - Holds all fixed metadata so that only per-CIR path data is read from disk
    // - Populate once via qrt_read_cache_init(), then pass to qrt_file_read()
    // - Plain data, no file handles, safe to copy and destroy
    struct qrt_read_cache
    {
        int version = 0;                // QRT file version
        unsigned n_orig = 0;            // Number of origin (TX) positions
        unsigned n_cir = 0;             // Number of CIRs per origin
        unsigned n_dest = 0;            // Number of destinations (RX)
        unsigned n_freq = 1;            // Number of frequency bands
        arma::fvec freq;                // Frequencies in Hz (converted from the stored unit), [n_freq]
        arma::fmat cir_pos;             // CIR positions, size [n_cir, 3]
        arma::fmat cir_orientation;     // CIR orientations (Euler), size [n_cir, 3]
        arma::fmat orig_pos_all;        // Origin positions, size [n_orig, 3]
        arma::fmat orig_orientation;    // Origin orientations (Euler), size [n_orig, 3]
        arma::u64_vec orig_index;       // Byte offsets from BOF to each origin data block, length [n_orig]
        arma::u64_vec path_data_offset; // Pre-computed absolute offset to path_data_index array per origin, length [n_orig]
        arma::fvec orig_freq;           // Version 4 only: frequency of each origin in Hz (0 for an unwritten origin), [n_orig]; empty otherwise
    };

    // Initialize a QRT read cache from a file
    qrt_read_cache qrt_read_cache_init(const std::string &fn,          // Path to the QRT file
                                       std::ifstream *file = nullptr); // Optional pre-opened ifstream

    // Initialize new QRT file and write metadata
    template <typename dtype>
    void qrt_file_init(
        const std::string &fn,                               // Path to the QRT file
        const arma::Col<dtype> &center_freq,                 // Center frequencies in [Hz], length [ n_freq ]
        const arma::Mat<dtype> &cir_pos,                     // CIR positions; [ n_cir, 3 ]
        const arma::Mat<dtype> &cir_orientation,             // CIR orientations; [ n_cir, 3 ]
        const std::vector<std::string> &dest_names = {"RX"}, // Receiver names; length [ n_rx ]
        const arma::u32_vec &cir_offset = {0},               // CIR offset for each receiver; length [ n_rx ]
        unsigned n_orig = 1,                                 // Number transmitters
        bool scalar_mode = false);                           // Flag for scalar layout

    // Append new TX path data to exisiting QRT file, throws when storage capacity is reached
    template <typename dtype>
    size_t qrt_file_append(
        const std::string &fn,                                         // Path to the QRT file
        const std::vector<quadriga_lib::path> &path_data,              // Path data
        const arma::Mat<dtype> &orig_pos,                              // Transmitter position in [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &orig_orientation = arma::Mat<dtype>(), // Transmitter orientation (bank, tilt, heading) in [rad], [3, 1] or [1, 3]; empty = zeros
        const std::string &orig_name = "TX");                          // Transmitter name

    // Read metadata from a QRT file
    void qrt_file_parse(
        const std::string &fn,                          // Path to the QRT file
        arma::uword *n_cir = nullptr,                   // Number of channel snapshots per origin point
        arma::uword *n_orig = nullptr,                  // Number of origin points (TX)
        arma::uword *n_dest = nullptr,                  // Number of destinations (RX)
        arma::uword *n_freq = nullptr,                  // Number of frequency bands
        arma::uvec *cir_offset = nullptr,               // CIR offset for each destination
        std::vector<std::string> *orig_names = nullptr, // Names of the origin points (TXs)
        std::vector<std::string> *dest_names = nullptr, // Names of the destination points (RXs)
        int *version = nullptr,                         // QRT file version
        arma::fvec *center_freq = nullptr,              // Center frequencies in [Hz], length [ n_freq ]
        arma::fmat *cir_pos = nullptr,                  // CIR positions in Cartesian coordinates, size [n_cir, 3]
        arma::fmat *cir_orientation = nullptr,          // CIR orientation in Euler angles in rad, size [n_cir, 3]
        arma::fmat *orig_pos = nullptr,                 // Origin (TX) positions, size [n_orig, 3]
        arma::fmat *orig_orientation = nullptr,         // Origin (TX) orientations, Euler angels in rad, size [n_orig, 3]
        std::ifstream *file = nullptr);                 // Optional pre-opened ifstream; if nullptr the file is opened from fn and closed on return

    // Read ray-tracing data of a list of CIRs from a QRT file
    // - Per-CIR outputs are vectors with one entry per requested CIR, positions and orientations are [3, n_out]
    // normalize_polmat options:
    //  0 : polmat as in QRT file, path_gain is -FSPL
    //  1 : polmat has a maximum column power of 1 (uplink: row power), path_gain is -FSPL - material losses
    template <typename dtype = double>
    void qrt_file_read(
        const std::string &fn,                                            // Path to the QRT file
        const arma::uvec &i_cir = {},                                     // CIR indices, 0-based, [n_out]; empty = all CIRs
        arma::uword i_orig = 0,                                           // Origin index (for downlink Origin = TX)
        bool uplink = false,                                              // Reverse direction: origin = RX, destination = TX
        arma::Col<dtype> *center_freq = nullptr,                          // Center frequency in [Hz], Size [n_freq]
        arma::Mat<dtype> *tx_pos = nullptr,                               // Transmitter positions in Cartesian coordinates in [m], [3, n_out]
        arma::Mat<dtype> *tx_orientation = nullptr,                       // Transmitter orientations (bank, tilt, heading) in [rad], [3, n_out]
        arma::Mat<dtype> *rx_pos = nullptr,                               // Receiver positions in Cartesian coordinates in [m], [3, n_out]
        arma::Mat<dtype> *rx_orientation = nullptr,                       // Receiver orientations (bank, tilt, heading) in [rad], [3, n_out]
        std::vector<arma::Mat<dtype>> *fbs_pos = nullptr,                 // First-bounce scatterer positions, [n_out] of [3, n_path]
        std::vector<arma::Mat<dtype>> *lbs_pos = nullptr,                 // Last-bounce scatterer positions, [n_out] of [3, n_path]
        std::vector<arma::Mat<dtype>> *path_gain = nullptr,               // Path gain (linear scale), [n_out] of [n_path, n_freq]
        std::vector<arma::Col<dtype>> *path_length = nullptr,             // Absolute path length from TX to RX phase center, [n_out] of [n_path]
        std::vector<arma::Cube<dtype>> *polmat = nullptr,                 // Polarization transfer matrix, [n_out] of [8, n_path, n_freq] or [2, n_path, n_freq] for v6 files
        std::vector<arma::Col<dtype>> *aod = nullptr,                     // Departure azimuth angles in [rad], [n_out] of [n_path]
        std::vector<arma::Col<dtype>> *eod = nullptr,                     // Departure elevation angles in [rad], [n_out] of [n_path]
        std::vector<arma::Col<dtype>> *aoa = nullptr,                     // Arrival azimuth angles in [rad], [n_out] of [n_path]
        std::vector<arma::Col<dtype>> *eoa = nullptr,                     // Arrival elevation angles in [rad], [n_out] of [n_path]
        std::vector<std::vector<arma::Mat<dtype>>> *path_coord = nullptr, // Interaction coordinates, [n_out] of [n_path] of [3, n_interact + 2]
        int normalize_polmat = 1,                                         // Normalize polmat options
        std::vector<arma::u32_vec> *n_interact = nullptr,                 // Number of mesh interactions per path, [n_out] of [n_path], 0=LOS
        std::vector<arma::fmat> *coord = nullptr,                         // Interaction coordinates, [n_out] of [3, sum(n_interact)]
        std::vector<std::vector<uint8_t>> *interact_type = nullptr,       // Interaction type codes, [n_out] of [sum(n_interact)]
        std::ifstream *file = nullptr,                                    // Optional pre-opened ifstream; if nullptr the file is opened from fn and closed on return
        const qrt_read_cache *cache = nullptr);                           // Optional pre-parsed metadata cache; without one, the whole header is read

    // Read raw ray-tracing data from QRT file
    std::vector<quadriga_lib::path> qrt_file_read_raw(
        const std::string &fn,                  // Path to the QRT file
        arma::uword i_orig = 0,                 // Origin index (for downlink Origin = TX)
        std::ifstream *file = nullptr,          // Optional pre-opened ifstream; if nullptr the file is opened from fn and closed on return
        const qrt_read_cache *cache = nullptr); // Optional pre-parsed metadata cache for maximum performance in tight loops

    // Path objects of one CIR from spherical path data, for qrt_file_append
    // - The path gain and the inverse free-space loss of the polyline length are folded into the stored Jones matrices
    // - LOS: 0 interactions; GR: 1 (reflection); NLOS: 2 (scattering)
    // - Use local coordinates (QRT stores float32): shift x and y only, the ground stays at z = 0
    std::vector<path> qrt_paths_from_scatterers(
        const arma::vec &center_freq,   // Frequencies in [Hz], [n_freq], 1 to 127 entries
        const arma::mat &tx_pos,        // Tx (origin) position in [m], [3, 1] or [1, 3]
        const arma::mat &rx_pos,        // Rx position of the CIR in [m], [3, 1] or [1, 3]
        const arma::mat &fbs_pos,       // First-bounce scatterer positions in [m], [3, n_path]
        const arma::mat &lbs_pos,       // Last-bounce scatterer positions in [m], [3, n_path]
        const arma::mat &path_gain,     // Path gains (linear), [n_path, n_freq]
        const arma::cube &polmat,       // Polarization transfer matrices, [8, n_path, n_freq]
        const arma::u32_vec &path_type, // Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_path]
        arma::uword i_cir = 0);         // CIR index, stored in path::i_cir, < 2^32

    // ---- Channel generation functions ----

    // Calculate channel coefficients for spherical waves
    // - Interpolates the transmit antenna pattern (including orientation and polarization)
    // - Interpolates the receive antenna pattern (including orientation and polarization)
    // - Calculates the channel response and delays in time domain for each MIMO sub-link (including antenna element coupling)
    // - LOS-Path identification: path-length must equal the TX-RX 3D-distance within 0.1 mm (use double precision for satellite links!)
    // - For the LOS-Path, FBS and LBS must be on the direct line between Tx and Rx (i.e. FBS-Pos = TX-Pos, LBS-Pos = RX-Pos)
    // - LOS-Path delays and amplitudes are calculated including the individual element positions
    // - If path length is shorter than the shortest possible path (TX > FBS > LBS > RX), shortest possible path is used for delay calculation
    // - If path length is longer (more interactions than FBS and LBS), it is used for the delay and phase of the phase centres;
    //   the extra length is added to the FBS-LBS segment, per-element offsets follow the FBS / LBS geometry
    // - Polarization transfer matrix: 2x2 Jones matrix (rows = RX pol. V/H, columns = TX pol. V/H), column-major, Re/Im
    //   interleaved: rows 0-1 = (V rx, V tx), 2-3 = (H rx, V tx), 4-5 = (V rx, H tx), 6-7 = (H rx, H tx)
    // - Polarization transfer matrix must be normalized (i.e., not include the free-space equivalent path gain)
    // - add_fake_los_path always adds one output path: slot 0 holds the LOS path (zero power if none was found); a LOS path
    //   found in the input leaves a zero-power path in its original slot
    template <typename dtype>
    void get_channels_spherical(
        const arrayant<dtype> &tx_array,        // Transmit array antenna with 'n_tx' elements (= ports after element coupling)
        const arrayant<dtype> &rx_array,        // Receive array antenna with 'n_rx' elements (= ports after element coupling)
        const arma::Mat<dtype> &tx_pos,         // Transmitter position in Cartesian coordinates [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &tx_orientation, // Transmitter orientation (bank, tilt, heading) in [rad], [3, 1] or [1, 3]
        const arma::Mat<dtype> &rx_pos,         // Receiver position in Cartesian coordinates [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &rx_orientation, // Receiver orientation (bank, tilt, heading) in [rad], [3, 1] or [1, 3]
        const arma::Mat<dtype> &fbs_pos,        // First-bounce scatterer positions, matrix of size [3, n_path]
        const arma::Mat<dtype> &lbs_pos,        // Last-bounce scatterer positions, matrix of size [3, n_path]
        const arma::Col<dtype> &path_gain,      // Path gain (linear scale), vector of length [n_path]
        const arma::Col<dtype> &path_length,    // Absolute path length from TX to RX phase center, vector of length [n_path]
        const arma::Mat<dtype> &polmat,         // Polarization transfer matrix, matrix of size [8, n_path]
        arma::Cube<dtype> &coeff_re,            // Output: Channel coefficients, real part, tensor of size [n_rx, n_tx, n_path]
        arma::Cube<dtype> &coeff_im,            // Output: Channel coefficients, imaginary part, tensor of size [n_rx, n_tx, n_path]
        arma::Cube<dtype> &delay,               // Output: Propagation delay in seconds, tensor of size [n_rx, n_tx, n_path]
        dtype center_freq = dtype(0.0),         // Center frequency in [Hz]; 0: coefficients at f = 0 (delay phase 1)
        bool use_absolute_delays = false,       // Option: If true, the LOS delay is included for all paths
        bool add_fake_los_path = false,         // Option: Add one path, slot 0 = LOS path (zero power if none was present)
        arma::Cube<dtype> *aod = nullptr,       // Optional output: Azimuth of Departure angles in [rad], Size [n_rx, n_tx, n_path]
        arma::Cube<dtype> *eod = nullptr,       // Optional output: Elevation of Departure angles in [rad], Size [n_rx, n_tx, n_path]
        arma::Cube<dtype> *aoa = nullptr,       // Optional output: Azimuth of Arrival angles in [rad], Size [n_rx, n_tx, n_path]
        arma::Cube<dtype> *eoa = nullptr,       // Optional output: Elevation of Arrival angles in [rad], Size [n_rx, n_tx, n_path]
        int use_kernel = 0);                    // Kernel of antenna interpolation and coefficients: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Calculate channel coefficients for spherical waves for several snapshots
    // - Calls get_channels_spherical for each snapshot, snapshots in parallel (OpenMP)
    // - Positions and orientations have one column per snapshot, or one column for all snapshots
    // - Output vectors get n_snap entries; entries that already have the output size are written in place
    // - The optional complex output is computed one snapshot per thread at a time (no split copy of all snapshots)
    template <typename dtype>
    void get_channels_spherical_vec(
        const arrayant<dtype> &tx_array,                               // Transmit array antenna with 'n_tx' elements (= ports after element coupling)
        const arrayant<dtype> &rx_array,                               // Receive array antenna with 'n_rx' elements (= ports after element coupling)
        const arma::Mat<dtype> &tx_pos,                                // Transmitter positions in Cartesian coordinates [m], [3, n_snap] or [3, 1]
        const arma::Mat<dtype> &tx_orientation,                        // Transmitter orientations (bank, tilt, heading) in [rad], [3, n_snap] or [3, 1]
        const arma::Mat<dtype> &rx_pos,                                // Receiver positions in Cartesian coordinates [m], [3, n_snap] or [3, 1]
        const arma::Mat<dtype> &rx_orientation,                        // Receiver orientations (bank, tilt, heading) in [rad], [3, n_snap] or [3, 1]
        const std::vector<arma::Mat<dtype>> &fbs_pos,                  // First-bounce scatterer positions, vector (n_snap) of matrices [3, n_path]
        const std::vector<arma::Mat<dtype>> &lbs_pos,                  // Last-bounce scatterer positions, vector (n_snap) of matrices [3, n_path]
        const std::vector<arma::Col<dtype>> &path_gain,                // Path gain (linear scale), vector (n_snap) of vectors [n_path]
        const std::vector<arma::Col<dtype>> &path_length,              // Absolute path length from TX to RX phase center, vector (n_snap) of vectors [n_path]
        const std::vector<arma::Mat<dtype>> &polmat,                   // Polarization transfer matrix, vector (n_snap) of matrices [8, n_path]
        std::vector<arma::Cube<dtype>> &coeff_re,                      // Output: Channel coefficients, real part, vector (n_snap) of tensors [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> &coeff_im,                      // Output: Channel coefficients, imaginary part, vector (n_snap) of tensors [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> &delay,                         // Output: Propagation delay in seconds, vector (n_snap) of tensors [n_rx, n_tx, n_path]
        dtype center_freq = dtype(0.0),                                // Center frequency in [Hz]; 0: coefficients at f = 0 (delay phase 1)
        bool use_absolute_delays = false,                              // Option: If true, the LOS delay is included for all paths
        bool add_fake_los_path = false,                                // Option: Add one path, slot 0 = LOS path (zero power if none was present)
        std::vector<arma::Cube<dtype>> *aod = nullptr,                 // Optional output: Azimuth of Departure angles in [rad], vector (n_snap) of tensors [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> *eod = nullptr,                 // Optional output: Elevation of Departure angles in [rad], vector (n_snap) of tensors [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> *aoa = nullptr,                 // Optional output: Azimuth of Arrival angles in [rad], vector (n_snap) of tensors [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> *eoa = nullptr,                 // Optional output: Elevation of Arrival angles in [rad], vector (n_snap) of tensors [n_rx, n_tx, n_path]
        std::vector<arma::Cube<std::complex<dtype>>> *coeff = nullptr, // Optional output: complex coeff_re / coeff_im (then left empty)
        int use_kernel = 0);                                           // Kernel of antenna interpolation and coefficients: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Calculate channel coefficients for spherical waves across multiple frequencies
    // - Extends get_channels_spherical to support frequency-dependent antenna patterns, path gains, and Jones matrices
    // - Geometry (angles, delays, LOS detection) is computed once and reused for all output frequencies
    // - TX/RX antenna patterns are interpolated from multi-frequency arrayant vectors at each output frequency
    // - Path gain and polarization transfer matrix are interpolated from freq to freq_new grids
    // - Supports both radio (speed of light) and acoustic (speed of sound) propagation
    // - Jones matrix polmat can have 2 rows (scalar pressure: Re, Im of (V rx, V tx) only) or 8 rows (full polarimetric, row
    //   order as in get_channels_spherical)
    // - NLOS path lengths and add_fake_los_path as in get_channels_spherical
    // - For baseband_freq_response_multi with remove_delay_phase = true, generate the channels with use_absolute_delays = true
    // - Output coefficients and delays are returned as vectors of cubes, one per output frequency
    template <typename dtype>
    void get_channels_multifreq(
        const std::vector<arrayant<dtype>> &tx_array,                  // Multi-frequency transmit array
        const std::vector<arrayant<dtype>> &rx_array,                  // Multi-frequency receive array
        const arma::Mat<dtype> &tx_pos,                                // Transmitter position [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &tx_orientation,                        // Transmitter orientation (bank, tilt, heading) [rad], [3, 1] or [1, 3]
        const arma::Mat<dtype> &rx_pos,                                // Receiver position [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &rx_orientation,                        // Receiver orientation (bank, tilt, heading) [rad], [3, 1] or [1, 3]
        const arma::Mat<dtype> &fbs_pos,                               // First-bounce scatterer positions [3, n_path]
        const arma::Mat<dtype> &lbs_pos,                               // Last-bounce scatterer positions [3, n_path]
        const arma::Mat<dtype> &path_gain,                             // Path gain (linear), [n_path, n_freq]
        const arma::Col<dtype> &path_length,                           // Absolute path length TX→RX [n_path]
        const arma::Cube<dtype> &polmat,                               // Polarization transfer matrix [8, n_path, n_freq] or [2, n_path, n_freq]
        const arma::Col<dtype> &freq,                                  // Input sample frequencies [Hz], length [n_freq]
        const arma::Col<dtype> &freq_new,                              // Target frequencies [Hz], length [n_freq_new]
        std::vector<arma::Cube<dtype>> &coeff_re,                      // Output: real part of coefficients, length [n_freq_new], each [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> &coeff_im,                      // Output: imag part of coefficients, length [n_freq_new], each [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> &delay,                         // Output: delays [s], length [n_freq_new], each [n_rx, n_tx, n_path]
        bool use_absolute_delays = false,                              // If true, include LOS delay in all paths
        bool add_fake_los_path = false,                                // Add one path, slot 0 = LOS path (zero power if none was present)
        dtype propagation_speed = 299792458.0,                         // Wave propagation speed [m/s]
        std::vector<arma::Cube<std::complex<dtype>>> *coeff = nullptr, // Optional output: complex coeff_re / coeff_im (then left empty)
        int use_kernel = 0);                                           // Kernel of the antenna interpolation: 0 = auto (GENERIC), 1 = GENERIC, 2 = AVX2

    // Calculate channels for Intelligent Reflective Surfaces (IRS)
    // - IRS is provided as a 3rd array antenna model comprised of 'n_irs' elements
    // - Requires 2 channel segments: (1) TX -> IRS and (2) IRS -> RX
    // - Assumes a low-rank channel where the IRS creates additional multipath components
    // - The IRS weights (phase shifts) are provided as Coupling matrix of the IRS array (column vectors); it must not be empty
    // - Multiple columns in the Coupling matrix can be seen as multiple IRS codebook entries. Only one entry can be selected
    // - Generates n_path_irs output paths where n_path_irs <= n_path_1 * n_path_2
    // - By default the 'irs_array' for both, the receive (TX-IRS) and receive (IRS-RX) direction.
    // - If different IRS patterns are required for the receive (IRS-RX) direction, a second IRS array can be supplied as optional input
    // - The optional output 'active_path_out' flags which of the n_path_1 * n_path_2 paths are in the output
    template <typename dtype> // Supported types: float or double
    void get_channels_irs(
        const arrayant<dtype> &tx_array,                            // Transmit array antenna with 'n_tx' elements (= ports after element coupling)
        const arrayant<dtype> &rx_array,                            // Receive array antenna with 'n_rx' elements (= ports after element coupling)
        const arrayant<dtype> &irs_array,                           // IRS array with 'n_irs' elements (only one port can be selected for channel generation)
        const arma::Mat<dtype> &tx_pos,                             // Transmitter position in Cartesian coordinates [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &tx_orientation,                     // Transmitter orientation (bank, tilt, heading) in [rad], [3, 1] or [1, 3]
        const arma::Mat<dtype> &rx_pos,                             // Receiver position in Cartesian coordinates [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &rx_orientation,                     // Receiver orientation (bank, tilt, heading) in [rad], [3, 1] or [1, 3]
        const arma::Mat<dtype> &irs_pos,                            // IRS position in Cartesian coordinates [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &irs_orientation,                    // IRS orientation (bank, tilt, heading) in [rad], [3, 1] or [1, 3]
        const arma::Mat<dtype> &fbs_pos_1,                          // First-bounce scatterer positions of TX-IRS paths, matrix of size [3, n_path_1]
        const arma::Mat<dtype> &lbs_pos_1,                          // Last-bounce scatterer positions of TX-IRS paths, matrix of size [3, n_path_1]
        const arma::Col<dtype> &path_gain_1,                        // Path gain (linear scale) of TX-IRS paths, vector of length [n_path_1]
        const arma::Col<dtype> &path_length_1,                      // Absolute path length from TX to IRS phase center, vector of length [n_path_1]
        const arma::Mat<dtype> &polmat_1,                           // Polarization transfer matrix of TX-IRS paths, matrix of size [8, n_path_1]
        const arma::Mat<dtype> &fbs_pos_2,                          // First-bounce scatterer positions of IRS-RX paths, matrix of size [3, n_path_2]
        const arma::Mat<dtype> &lbs_pos_2,                          // Last-bounce scatterer positions of IRS-RX paths, matrix of size [3, n_path_2]
        const arma::Col<dtype> &path_gain_2,                        // Path gain (linear scale) of IRS-RX paths, vector of length [n_path_2]
        const arma::Col<dtype> &path_length_2,                      // Absolute path length from IRS to RX phase center, vector of length [n_path_2]
        const arma::Mat<dtype> &polmat_2,                           // Polarization transfer matrix of IRS-RX paths, matrix of size [8, n_path_2]
        arma::Cube<dtype> &coeff_re,                                // Output: Channel coefficients, real part, tensor of size [n_rx, n_tx, n_path_irs]
        arma::Cube<dtype> &coeff_im,                                // Output: Channel coefficients, imaginary part, tensor of size [n_rx, n_tx, n_path_irs]
        arma::Cube<dtype> &delay,                                   // Output: Propagation delay in seconds, tensor of size [n_rx, n_tx, n_path_irs]
        arma::uword i_irs = 0,                                      // IRS codebook entry (= port number in the IRS array)
        dtype threshold_dB = -140.0,                                // Threshold in dB relative to path gain below which paths are discarded from the output
        dtype center_freq = 0.0,                                    // Center frequency in [Hz]; 0: coefficients at f = 0 (delay phase 1)
        bool use_absolute_delays = false,                           // Option: If true, the LOS delay is included for all paths
        arma::Cube<dtype> *aod = nullptr,                           // Optional output: Azimuth of Departure angles in [rad], Size [n_rx, n_tx, n_path_irs]
        arma::Cube<dtype> *eod = nullptr,                           // Optional output: Elevation of Departure angles in [rad], Size [n_rx, n_tx, n_path_irs]
        arma::Cube<dtype> *aoa = nullptr,                           // Optional output: Azimuth of Arrival angles in [rad], Size [n_rx, n_tx, n_path_irs]
        arma::Cube<dtype> *eoa = nullptr,                           // Optional output: Elevation of Arrival angles in [rad], Size [n_rx, n_tx, n_path_irs]
        const arrayant<dtype> &irs_array_2 = arrayant<dtype>(),     // Optional input: Transmit IRS for the second segment with 'n_irs' elements; empty = irs_array
        const std::vector<bool> &active_path = std::vector<bool>(), // Optional input: Switch to activate individual paths, ignores threshold_dB
        std::vector<bool> *active_path_out = nullptr,               // Optional output: Paths in the output, [n_path_1 * n_path_2]
        int use_kernel = 0);                                        // Kernel of both segments (get_channels_spherical): 0 = auto (GENERIC), 1 = GENERIC, 2 = AVX2

    // Calculate channel coefficients for planar waves
    // - Interpolates the transmit antenna pattern (including orientation and polarization)
    // - Interpolates the receive antenna pattern (including orientation and polarization)
    // - Calculates the channel response and delays in time domain for each MIMO sub-link (including antenna element coupling)
    // - LOS-Path identification: path-length must equal the TX-RX 3D-distance within 0.1 mm
    // - Angles are ignored for LOS path identification
    // - Polarization transfer matrix: 2x2 Jones matrix (rows = RX pol. V/H, columns = TX pol. V/H), column-major, Re/Im
    //   interleaved: rows 0-1 = (V rx, V tx), 2-3 = (H rx, V tx), 4-5 = (V rx, H tx), 6-7 = (H rx, H tx)
    // - Polarization transfer matrix must be normalized (i.e., not include the path gain)
    // - add_fake_los_path always adds one output path: slot 0 holds the strongest LOS path (zero power if none was found);
    //   a LOS path found in the input leaves a zero-power path in its original slot
    // - Option to calculate the Doppler weights from orientation (+1 Moves towards path, -1 moves away from path)
    template <typename dtype>
    void get_channels_planar(
        const arrayant<dtype> &tx_array,        // Transmit array antenna with 'n_tx' elements (= ports after element coupling)
        const arrayant<dtype> &rx_array,        // Receive array antenna with 'n_rx' elements (= ports after element coupling)
        const arma::Mat<dtype> &tx_pos,         // Transmitter position in Cartesian coordinates [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &tx_orientation, // Transmitter orientation (bank, tilt, heading) in [rad], [3, 1] or [1, 3]
        const arma::Mat<dtype> &rx_pos,         // Receiver position in Cartesian coordinates [m], [3, 1] or [1, 3]
        const arma::Mat<dtype> &rx_orientation, // Receiver orientation (bank, tilt, heading) in [rad], [3, 1] or [1, 3]
        const arma::Col<dtype> &aod,            // Departure azimuth angles in [rad], vector of length 'n_path'
        const arma::Col<dtype> &eod,            // Departure elevation angles in [rad], vector of length 'n_path'
        const arma::Col<dtype> &aoa,            // Arrival azimuth angles in [rad], vector of length 'n_path'
        const arma::Col<dtype> &eoa,            // Arrival elevation angles in [rad], vector of length 'n_path'
        const arma::Col<dtype> &path_gain,      // Path gain (linear scale), vector of length [n_path]
        const arma::Col<dtype> &path_length,    // Absolute path length from TX to RX phase center, vector of length [n_path]
        const arma::Mat<dtype> &polmat,         // Polarization transfer matrix, matrix of size [8, n_path]
        arma::Cube<dtype> &coeff_re,            // Output: Channel coefficients, real part, tensor of size [n_rx, n_tx, n_path(+1)]
        arma::Cube<dtype> &coeff_im,            // Output: Channel coefficients, imaginary part, tensor of size [n_rx, n_tx, n_path(+1)]
        arma::Cube<dtype> &delay,               // Output: Propagation delay in seconds, tensor of size [n_rx, n_tx, n_path(+1)]
        dtype center_freq = dtype(0.0),         // Center frequency in [Hz]; 0: coefficients at f = 0 (delay phase 1)
        bool use_absolute_delays = false,       // Option: If true, the LOS delay is included for all paths
        bool add_fake_los_path = false,         // Option: Add one path, slot 0 = LOS path (zero power if none was present)
        arma::Col<dtype> *rx_Doppler = nullptr, // Optional output: Doppler weights for moving RX, vector of length 'n_path(+1)'
        int use_kernel = 0);                    // Kernel of the antenna interpolation: 0 = auto (GENERIC), 1 = GENERIC, 2 = AVX2

    // Calculate channel coefficients for planar waves for several snapshots
    // - Calls get_channels_planar for each snapshot, snapshots in parallel (OpenMP)
    // - Positions and orientations have one column per snapshot, or one column for all snapshots
    // - Output vectors get n_snap entries; entries that already have the output size are written in place
    // - The optional complex output is computed one snapshot per thread at a time (no split copy of all snapshots)
    template <typename dtype>
    void get_channels_planar_vec(
        const arrayant<dtype> &tx_array,                               // Transmit array antenna with 'n_tx' elements (= ports after element coupling)
        const arrayant<dtype> &rx_array,                               // Receive array antenna with 'n_rx' elements (= ports after element coupling)
        const arma::Mat<dtype> &tx_pos,                                // Transmitter positions in Cartesian coordinates [m], [3, n_snap] or [3, 1]
        const arma::Mat<dtype> &tx_orientation,                        // Transmitter orientations (bank, tilt, heading) in [rad], [3, n_snap] or [3, 1]
        const arma::Mat<dtype> &rx_pos,                                // Receiver positions in Cartesian coordinates [m], [3, n_snap] or [3, 1]
        const arma::Mat<dtype> &rx_orientation,                        // Receiver orientations (bank, tilt, heading) in [rad], [3, n_snap] or [3, 1]
        const std::vector<arma::Col<dtype>> &aod,                      // Departure azimuth angles in [rad], vector (n_snap) of vectors [n_path]
        const std::vector<arma::Col<dtype>> &eod,                      // Departure elevation angles in [rad], vector (n_snap) of vectors [n_path]
        const std::vector<arma::Col<dtype>> &aoa,                      // Arrival azimuth angles in [rad], vector (n_snap) of vectors [n_path]
        const std::vector<arma::Col<dtype>> &eoa,                      // Arrival elevation angles in [rad], vector (n_snap) of vectors [n_path]
        const std::vector<arma::Col<dtype>> &path_gain,                // Path gain (linear scale), vector (n_snap) of vectors [n_path]
        const std::vector<arma::Col<dtype>> &path_length,              // Absolute path length from TX to RX phase center, vector (n_snap) of vectors [n_path]
        const std::vector<arma::Mat<dtype>> &polmat,                   // Polarization transfer matrix, vector (n_snap) of matrices [8, n_path]
        std::vector<arma::Cube<dtype>> &coeff_re,                      // Output: Channel coefficients, real part, vector (n_snap) of tensors [n_rx, n_tx, n_path(+1)]
        std::vector<arma::Cube<dtype>> &coeff_im,                      // Output: Channel coefficients, imaginary part, vector (n_snap) of tensors [n_rx, n_tx, n_path(+1)]
        std::vector<arma::Cube<dtype>> &delay,                         // Output: Propagation delay in seconds, vector (n_snap) of tensors [n_rx, n_tx, n_path(+1)]
        dtype center_freq = dtype(0.0),                                // Center frequency in [Hz]; 0: coefficients at f = 0 (delay phase 1)
        bool use_absolute_delays = false,                              // Option: If true, the LOS delay is included for all paths
        bool add_fake_los_path = false,                                // Option: Add one path, slot 0 = LOS path (zero power if none was present)
        std::vector<arma::Col<dtype>> *rx_Doppler = nullptr,           // Optional output: Doppler weights for moving RX, vector (n_snap) of vectors [n_path(+1)]
        std::vector<arma::Cube<std::complex<dtype>>> *coeff = nullptr, // Optional output: complex coeff_re / coeff_im (then left empty)
        int use_kernel = 0);                                           // Kernel of the antenna interpolation: 0 = auto (GENERIC), 1 = GENERIC, 2 = AVX2

    // Channel generation function for IEEE TGn, TGac, TGax and TGah indoor channel models
    // - Returns vector of channel objects, length n_users
    // - Depends on arrayant and channel classes
    // - 2D model, no elevation angles
    std::vector<channel<double>> get_channels_ieee_indoor(
        const arrayant<double> &ap_array,                 // Access point array antenna with 'n_tx' elements (= ports after element coupling)
        const arrayant<double> &sta_array,                // Mobile station array antenna with 'n_rx' elements (= ports after element coupling)
        const std::string &channel_type,                  // Channel Model Type (A, B, C, D, E, F) as defined by TGn
        double center_freq = 5.25e9,                      // Carrier frequency in Hz
        double tap_spacing = 10.0e-9,                     // Taps spacing in seconds, must be equal to 10 ns divided by a power of 2, TGn = 10e-9
        arma::uword n_users = 1,                          // Number of user (only for TGac, TGah)
        double observation_time = 0.0,                    // Channel observation time in seconds (0.0 = static channel)
        double update_interval = 1.0e-3,                  // Channel update interval in seconds
        double speed_station_kmh = 0.0,                   // Movement speed of the station in km/h (optional feature, default = 0), movement direction = AoA_offset
        double speed_env_kmh = 1.2,                       // Movement speed of the environment in km/h (default = 1.2 for TGn) use 0.089 for TGac
        const arma::vec &dist = arma::vec(),              // Distance between TX and RX in meters, [n_users] or [1]; empty = 4.99 m
        const arma::uvec &n_floors = arma::uvec(),        // Number of floors per user (TGah / TGax), up to 4, [n_users] or [1]; empty = 0; attenuation only, no delay
        bool uplink = false,                              // Default channel direction is downlink, set uplink to true to get reverse direction (transpose, same Doppler shift)
        const arma::mat &offset_angles_deg = arma::mat(), // Offset angles in degree for MU-MIMO channels, [4, n_users], rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS; empty = TGac auto
        arma::uword n_subpath = 20,                       // Number of sub-paths per path and cluster for Laplacian AS mapping
        double doppler_effect = 50.0,                     // Special Doppler effects in models D, E (fluorescent lights, value = mains freq. in Hz) and F (moving vehicle speed in km/h), >= 0, 0 = off
        arma::sword seed = -1,                            // Numeric seed, optional, value -1 disabled seed and uses system random device
        double kf = NAN,                                  // Overrides the default KF (linear scale); NAN = default, else finite and >= 0
        double xpr_nlos = NAN,                            // Overrides the default cross-polarization ratio (linear scale) for NLOS paths; NAN = default, else finite and > 0
        double sf_std_los_dB = NAN,                       // Overrides the default shadow fading STD for LOS channels in dB; NAN = default, else finite and >= 0
        double sf_std_nlos_dB = NAN,                      // Overrides the default shadow fading STD for NLOS channels in dB; NAN = default, else finite and >= 0
        double dist_breakpoint = NAN,                     // Overrides the default breakpoint distance in meters; NAN = default, else finite and > 0
        const arma::uvec &n_walls = arma::uvec(),         // Number of walls per user (TGax), [n_users] or [1]; empty = 0
        double wall_loss_dB = 5.0);                       // Penetration loss for a single wall in dB; TGax defines 5.0 (default) or 7.0

}

#endif
