// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (http://quadriga-lib.org)
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

#ifndef quadriga_channel_H
#define quadriga_channel_H

#include <armadillo>
#include <string>
#include <vector>
#include <any>
#include <cstring>
#include <cmath>
#include <complex>

namespace quadriga_lib
{
    // Class for storing and managing channel data (+ metadata)
    // Note: "n_path" is different for each snapshot
    template <typename dtype> // float or double
    class channel
    {
    public:
        std::string name = "empty";                      // Name of the channel object
        arma::Col<dtype> center_frequency;               // Center frequency in [Hz], vector of length [1] or [n_snap] or []
        arma::Mat<dtype> tx_pos;                         // Transmitter positions, matrix of size [3, n_snap] or [3, 1]
        arma::Mat<dtype> rx_pos;                         // Receiver positions, matrix of size [3, n_snap] or [3, 1]
        arma::Mat<dtype> tx_orientation;                 // Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
        arma::Mat<dtype> rx_orientation;                 // Receiver orientation, matrix of size [3, n_snap] or [3, 1] or []
        std::vector<arma::Cube<dtype>> coeff_re;         // Channel coefficients, real part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> coeff_im;         // Channel coefficients, imaginary part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> delay;            // Path delays in seconds, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path] or [1, 1, n_path]
        std::vector<arma::Col<dtype>> path_gain;         // Path gain before antenna patterns, vector (n_snap) of vectors of length [n_path]
        std::vector<arma::Col<dtype>> path_length;       // Absolute path length from TX to RX phase center, vector (n_snap) of vectors of length [n_path]
        std::vector<arma::Mat<dtype>> path_polarization; // Polarization transfer function, vector (n_snap) of matrices of size [8, n_path], interleaved complex
        std::vector<arma::Mat<dtype>> path_angles;       // Departure and arrival angles, vector (n_snap) of matrices of size [n_path, 4], {AOD, EOD, AOA, EOA}
        std::vector<arma::Mat<dtype>> path_fbs_pos;      // First-bounce scatterer positions, matrices of size [3, n_path]
        std::vector<arma::Mat<dtype>> path_lbs_pos;      // Last-bounce scatterer positions, matrices of size [3, n_path]
        std::vector<arma::Col<unsigned>> no_interact;    // Number interaction points of a path with the environment, 0 = LOS, vector (n_snap) of vectors of length [n_path]
        std::vector<arma::Mat<dtype>> interact_coord;    // Interaction coordinates of paths with the environment, matrices of size [3, sum(no_interact)]
        std::vector<std::string> par_names;              // Names of unstructured data fields
        std::vector<std::any> par_data;                  // Unstructured data of types {string, float, double, int, long int, arma::Col<dtype>, arma::Mat<dtype>, arma::Cube<dtype>}
        int initial_position = 0;                        // Index of reference position, values between 0 and n_snap-1 (mainly used internally)
        channel() {};                                    // Default constructor

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
        void add_paths(arma::uword i_snap,                                      // Snapshot index to which the paths should be added
                       const arma::Cube<dtype> *coeff_re_add = nullptr,         // Channel coefficients, real part, Cube of size [n_rx, n_tx, n_path_add]
                       const arma::Cube<dtype> *coeff_im_add = nullptr,         // Channel coefficients, imaginary part, Cube of size [n_rx, n_tx, n_path_add]
                       const arma::Cube<dtype> *delay_add = nullptr,            // Path delays in seconds, Cube of size [n_rx, n_tx, n_path_add] or [1, 1, n_path_add]
                       const arma::u32_vec *no_interact_add = nullptr,          // Number of interaction coordinates, vector of length [n_path_add]
                       const arma::Mat<dtype> *interact_coord_add = nullptr,    // Interaction coordinates, matrix of size [3, sum(no_interact)]
                       const arma::Col<dtype> *path_gain_add = nullptr,         // Path gain before antenna patterns, vector of length [n_path_add]
                       const arma::Col<dtype> *path_length_add = nullptr,       // Absolute path length from TX to RX phase center, vector of length [n_path_add]
                       const arma::Mat<dtype> *path_polarization_add = nullptr, // Polarization transfer function, Matrix of size [8, n_path_add], interleaved complex
                       const arma::Mat<dtype> *path_angles_add = nullptr,       // Departure and arrival angles, Matrix of size [n_path_add, 4], {AOD, EOD, AOA, EOA}
                       const arma::Mat<dtype> *path_fbs_pos_add = nullptr,      // First-bounce scatterer positions, Matrix of size [3, n_path_add]
                       const arma::Mat<dtype> *path_lbs_pos_add = nullptr);     // Last-bounce scatterer positions, Matrix of size [3, n_path_add]

        // Calculate the the effective path gain (linear scale)
        // - sum up the power of all paths and average over the transmit and receive antennas
        // - if coeff_re/im are not available, use path_gain/polarization instead (assume ideal XPOL antennas)
        // - throws an error if neither coeff_re/im nor path_polarization is available
        // - returns one value for each snapshot (vector with n_snap elements)
        arma::Col<dtype> calc_effective_path_gain(bool assume_valid = false) const;

        // Write propagation paths to a Wavefront OBJ file
        // - e.g. for visualization in Blender
        void write_paths_to_obj_file(std::string fn,                 // Filename of the OBJ file
                                     arma::uword max_no_paths = 0,   // Maximum number of paths to be shown, Default: all
                                     dtype gain_max = -60.0,         // Maximum path gain in dB (only for color-coding)
                                     dtype gain_min = -140.0,        // Minimum path gain in dB (only for color-coding and path selection)
                                     std::string colormap = "jet",   // Colormap for the visualization
                                     arma::uvec i_snap = {},         // Snapshot indices, 0-based, empty = export all
                                     dtype radius_max = 0.05,        // Maximum tube radius in meters
                                     dtype radius_min = 0.01,        // Minimum tube radius in meters
                                     arma::uword n_edges = 5) const; // Number of vertices in the circle building the tube, must be >= 3
    };

    // Function to obtain the HDF5 library version
    std::string get_HDF5_version();

    // Returns type ID of a std::any field and allows low-level data access
    // - Optional parameter dims = Pointer to 3-element array to store the size of each dimension
    // - For strings, dims[0] contains the length of the string
    // - Optional parameter dataptr = Raw pointer to the internal data array
    // - WARNING: Use  dataptr with caution. It allows full RW-access to the internal data, ignores const and is not typesafe!
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
    int any_type_id(const std::any *data, unsigned long long *dims = nullptr, void **dataptr = nullptr);

    // Create a new channel HDF file and set the index to given storage layout
    // - Channels can be structured in a multi-dimensional array (up to 4D)
    //   For instance, the first dimension might represent the Base Station (BS), the second the
    //   User Equipment (UE), and the third the frequency.
    // - Dimensions of the storage layout must be defined when the file is initially created
    // - It is possible to reshape the layout using 'hdf5_reshape_layout', but the total number of
    //   entries must not change
    void hdf5_create(std::string fn,      // Filename of the HDF5 file
                     unsigned nx = 65536, // Number of x-entries in the storage layout (first dimension, e.g. for BSs)
                     unsigned ny = 1,     // Number of y-entries in the storage layout (second dimension, e.g. for UEs)
                     unsigned nz = 1,     // Number of z-entries in the storage layout (second dimension, e.g. for carrier frequency)
                     unsigned nw = 1);    // Number of w-entries in the storage layout (fourth dimension)

    // Save data to HDF file (stand alone function)
    // - All data is stored in single precision
    // - Data is added to the end of the file
    // - The index is updated to include to the newly written data
    // - If file does not exist, a new file will be created with (nx = 65535, ny = 1, nz = 1, nw = 1)
    // - Returns 0 if new dataset was created, 1 if dataset was overwritten or modified
    // - The switch "assume_valid" can be used to skip the data integrity check (for better performance)
    // - Throws and error if the requested index in the file has not been reserved during hdf5_create
    template <typename dtype>                              // Float or double
    int hdf5_write(const quadriga_lib::channel<dtype> *ch, // Channel object to write
                   std::string fn,                         // Filename of the HDF5 file
                   unsigned ix = 0,                        // x-Index in the HDF file
                   unsigned iy = 0,                        // y-Index in the HDF file
                   unsigned iz = 0,                        // z-Index in the HDF file
                   unsigned iw = 0,                        // w-Index in the HDF file
                   bool assume_valid = false);             // Swith to skip the data integrity check

    // Read the storage layout of channel data inside an HDF5 file
    // - Returns 4-element vector with storage layout in form: {nx, ny, nz, nw}
    // - If file does not exist, output will be {0,0,0,0}, no error-message
    // - Optional output: ChannelID = vector containing the ChannelID in the file, serialized, length nx × ny × nz × nw
    // - channelIDs with value 0 indicate that there is no data stored at this index
    arma::u32_vec hdf5_read_layout(std::string fn, arma::u32_vec *channelID = nullptr);

    // Reshapes the storage layout inside an existing HDF5 file
    // - Total number of elements in the HDF5 file is fixed when calling `hdf5_create`
    // - Reshaping is possible only when the total number does not change
    void hdf5_reshape_layout(std::string fn,   // Filename of the HDF5 file
                             unsigned nx,      // Number of x-entries in the storage layout (first dimension, e.g. for BSs)
                             unsigned ny = 1,  // Number of y-entries in the storage layout (second dimension, e.g. for UEs)
                             unsigned nz = 1,  // Number of z-entries in the storage layout (second dimension, e.g. for carrier frequency)
                             unsigned nw = 1); // Number of w-entries in the storage layout (fourth dimension)

    // Read channel object from HDF5 file
    // - Returns empty channel object if channel ID does not exist
    // - All structured data is stored in single precision, but is converted to double in case if dtype = double
    // - Conversion to float or double is not done for unstructured data
    template <typename dtype>                          // Read as float or double
    channel<dtype> hdf5_read_channel(std::string fn,   // Filename of the HDF5 file
                                     unsigned ix = 0,  // x-Index in the HDF file
                                     unsigned iy = 0,  // y-Index in the HDF file
                                     unsigned iz = 0,  // z-Index in the HDF file
                                     unsigned iw = 0); // w-Index in the HDF file

    // Read single unstructured dataset from HDF5 file
    // - Besides structured channel data, extra datasets of various types can be added
    // - They are identified by a prefix (usually "par_") followed by a dataset name (par_name)
    // - Returns a std::any object, that can be further analyzed using the function quadriga_lib::any_type_id
    // - Returns empty std::any if there is no dataset at this location or under this name
    std::any hdf5_read_dset(std::string fn,               // Filename of the HDF5 file
                            std::string par_name,         // Dataset name without the prefix (typically "par_")
                            unsigned ix = 0,              // x-Index in the HDF file
                            unsigned iy = 0,              // y-Index in the HDF file
                            unsigned iz = 0,              // z-Index in the HDF file
                            unsigned iw = 0,              // w-Index in the HDF file
                            std::string prefix = "par_"); // Option to set a different prefix

    // Read names of the unstructured datasets from the HDF file
    // - Returns number of the unstructured datasets at the given location
    // - par_names will contain all dataset name without the prefix
    arma::uword hdf5_read_dset_names(std::string fn,                      // Filename of the HDF5 file
                                     std::vector<std::string> *par_names, // Output data
                                     unsigned ix = 0,                     // x-Index in the HDF file
                                     unsigned iy = 0,                     // y-Index in the HDF file
                                     unsigned iz = 0,                     // z-Index in the HDF file
                                     unsigned iw = 0,                     // w-Index in the HDF file
                                     std::string prefix = "par_");        // Option to set a different prefix

    // Write single unstructured data field to HDF5 file
    // - Scalar types: string, unsigned, int, long long, unsigned long long, float, double
    // - Armadillo Vectors, Matrices and Cubes with types: unsigned, int, unsigned long long, sword, float, double
    // - Vector types: Armadillo Row and Col types are mapped to 1D storage type
    // - arma::Row will be converted to arma::Col
    // - Parameter name may only contain letters and numbers and the underscore "_"
    // - An additional prefix "par_" will be added to the name before writing to the HDF file
    // - Unsupported data types will cause an error
    void hdf5_write_dset(std::string fn,               // Filename of the HDF5 file
                         std::string par_name,         // Dataset name without the prefix (typically "par_")
                         const std::any *par_data,     // Data to be written
                         unsigned ix = 0,              // x-Index in the HDF file
                         unsigned iy = 0,              // y-Index in the HDF file
                         unsigned iz = 0,              // z-Index in the HDF file
                         unsigned iw = 0,              // w-Index in the HDF file
                         std::string prefix = "par_"); // Option to set a different prefix

    // Compute the baseband frequency response of a MIMO channel
    // - Inputs are given in the time domain
    // - Calculates the discrete Fourier transform of the coefficients
    // - Outputs are returned in the frequency domain at the given sample positions
    // - Uses AVX2 to accelerate the computations (calculate 8 carriers in parallel)
    // - It is possible to call this function in a loop for multiple snapshots and parallelize this loop using OpenMP
    // - Internal calculations are done in single precision
    template <typename dtype>
    void baseband_freq_response(const arma::Cube<dtype> *coeff_re,                // Channel coefficients, real part, cube of size [n_rx, n_tx, n_path]
                                const arma::Cube<dtype> *coeff_im,                // Channel coefficients, imaginary part, cube of size [n_rx, n_tx, n_path]
                                const arma::Cube<dtype> *delay,                   // Path delays in seconds, cube of size [n_rx, n_tx, n_path] or [1, 1, n_path]
                                const arma::Col<dtype> *pilot_grid,               // Sub-carrier positions, relative to the bandwidth, 0.0 = fc, 1.0 = fc+bandwidth, Size: [ n_carriers ]
                                const double bandwidth,                           // The baseband bandwidth in [Hz]
                                arma::Cube<dtype> *hmat_re = nullptr,             // Output: Channel matrix (H), real part, Size [n_rx, n_tx, n_carriers]
                                arma::Cube<dtype> *hmat_im = nullptr,             // Output: Channel matrix (H), imaginary part, Size [n_rx, n_tx, n_carriers]
                                arma::Cube<std::complex<dtype>> *hmat = nullptr); // Output: Channel matrix (H), complex-valued, Size [n_rx, n_tx, n_carriers]

    // Compute the baseband frequency response of multiple MIMO channels
    // - Wrapper function for "quadriga_lib::baseband_freq_response" to process multiple snapshot at once
    // - Optional input i_snap can be used to select a subset of the given coeff_re, coeff_im, delay
    // - Uses OpenMP to process snapshots in parallel
    template <typename dtype>
    void baseband_freq_response_vec(const std::vector<arma::Cube<dtype>> *coeff_re, // Channel coefficients, real part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
                                    const std::vector<arma::Cube<dtype>> *coeff_im, // Channel coefficients, imaginary part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
                                    const std::vector<arma::Cube<dtype>> *delay,    // Path delays in seconds, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path] or [1, 1, n_path]
                                    const arma::Col<dtype> *pilot_grid,             // Sub-carrier positions, relative to the bandwidth, 0.0 = fc, 1.0 = fc+bandwidth, Size: [ n_carriers ]
                                    const double bandwidth,                         // The baseband bandwidth in [Hz]
                                    std::vector<arma::Cube<dtype>> *hmat_re,        // Output: Channel matrices (H), real part, vector (n_out) of Cubes of size [n_rx, n_tx, n_carriers]
                                    std::vector<arma::Cube<dtype>> *hmat_im,        // Output: Channel matrices (H), imaginary part, vector (n_out) of Cubes of size [n_rx, n_tx, n_carriers]
                                    const arma::u32_vec *i_snap = nullptr);         // Snapshot indices, 0-based, optional input, vector of length "n_out"

    // Read metadata from a QRT file
    void qrt_file_parse(const std::string &fn,                           // Path to the QRT file
                        arma::uword *no_cir = nullptr,                   // Number of channel snapshots per origin point
                        arma::uword *no_orig = nullptr,                  // Number of origin points (TX)
                        arma::uword *no_dest = nullptr,                  // Number of destinations (RX)
                        arma::uvec *cir_offset = nullptr,                // CIR offset for each destination
                        std::vector<std::string> *orig_names = nullptr,  // Names of the origin points (TXs)
                        std::vector<std::string> *dest_names = nullptr); // Names of the destination points (RXs)

    // Read ray-tracing data from QRT file
    template <typename dtype>
    void qrt_file_read(const std::string &fn,                                // Path to the QRT file
                       arma::uword i_cir = 0,                                // Snapshot index
                       arma::uword i_orig = 0,                               // Origin index (for downlink Origin = TX)
                       bool downlink = true,                                 // Switch for uplink / downlink direction
                       dtype *center_frequency = nullptr,                    // Center frequency in [Hz]; a value of 0 disables phase calculation in coefficients
                       arma::Col<dtype> *tx_pos = nullptr,                   // Transmitter position in Cartesian coordinates, Size [3, 1]
                       arma::Col<dtype> *tx_orientation = nullptr,           // Transmitter orientation (bank, tilt, head) in [rad], Size [3, 1]
                       arma::Col<dtype> *rx_pos = nullptr,                   // Receiver position in Cartesian coordinates, Size [3, 1]
                       arma::Col<dtype> *rx_orientation = nullptr,           // Receiver orientation (bank, tilt, head) in [rad], Size [3, 1]
                       arma::Mat<dtype> *fbs_pos = nullptr,                  // First-bounce scatterer positions, matrix of size [3, n_path]
                       arma::Mat<dtype> *lbs_pos = nullptr,                  // Last-bounce scatterer positions, matrix of size [3, n_path]
                       arma::Col<dtype> *path_gain = nullptr,                // Path gain (linear scale), vector of length [n_path]
                       arma::Col<dtype> *path_length = nullptr,              // Absolute path length from TX to RX phase center, vector of length [n_path]
                       arma::Mat<dtype> *M = nullptr,                        // Polarization transfer matrix, matrix of size [8, n_path]
                       arma::Col<dtype> *aod = nullptr,                      // Departure azimuth angles in [rad], vector of length 'n_path'
                       arma::Col<dtype> *eod = nullptr,                      // Departure elevation angles in [rad], vector of length 'n_path'
                       arma::Col<dtype> *aoa = nullptr,                      // Arrival azimuth angles in [rad], vector of length 'n_path'
                       arma::Col<dtype> *eoa = nullptr,                      // Arrival elevation angles in [rad], vector of length 'n_path'
                       std::vector<arma::Mat<dtype>> *path_coord = nullptr); // Interaction coordinates, vector (n_path) of matrices of size [3, n_interact + 2]

}

#endif
