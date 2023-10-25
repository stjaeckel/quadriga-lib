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

#ifndef quadriga_channel_H
#define quadriga_channel_H

#include <armadillo>
#include <string>
#include <vector>
#include <any>

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
        arma::Mat<dtype> rx_pos;                         // Receiver positions, matrix of size [3, n_snap]
        arma::Mat<dtype> tx_orientation;                 // Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
        arma::Mat<dtype> rx_orientation;                 // Receiver orientation, matrix of size [3, n_snap] or [3, 1] or []
        std::vector<arma::Cube<dtype>> coeff_re;         // Channel coefficients, real part, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> coeff_im;         // Channel coefficients, imaginary part, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
        std::vector<arma::Cube<dtype>> delay;            // Path delays in seconds, vector (n_snap) of tensors of size [n_rx, n_tx, n_path] or [1, 1, n_path]
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
        channel(){};                                     // Default constructor

        unsigned long long n_snap() const; // Number of snapshots
        unsigned long long n_rx() const;   // Number of receive antennas in coefficient matrix, returns 0 if there are no coefficients
        unsigned long long n_tx() const;   // Number of transmit antennas in coefficient matrix, returns 0 if there are no coefficients
        arma::uvec n_path() const;         // Number of paths per snapshot (unsigned long long)
        bool empty() const;                // Returns true if the channel object contains no structured data

        // Validate integrity
        std::string is_valid() const; // Returns an empty string if channel object is valid or an error message otherwise

        // Save data to HDF file
        // - All data is stored in single precision
        // - Data is added to the end of the file
        // - The index is updated to include to the newly written data
        // - If file does not exist, a new file will be created with (nx = 65535, ny = 1, nz = 1, nw = 1)
        // - Returns 0 if new dataset was created, 1 if dataset was overwritten or modified
        // - The switch "assume_valid" can be used to skip the data integrity check (for better performance)
        int hdf5_write(std::string fn, unsigned ix = 0, unsigned iy = 0, unsigned iz = 0, unsigned iw = 0, bool assume_valid = false) const;
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
    void hdf5_create(std::string fn, unsigned nx = 65536, unsigned ny = 1, unsigned nz = 1, unsigned nw = 1);

    // Read storage layout from HDF file
    // - Output will be a 4-element vector
    // - If file does not exist, output will be [0,0,0,0], no error-message
    // - Optional output: has_value = vector containing the ChannelID in the file
    // - channelIDs with value 0 indicate that there is no data stored at this index
    arma::Col<unsigned> hdf5_read_layout(std::string fn, arma::Col<unsigned> *channelID = nullptr);

    // Reshape storage layout
    void hdf5_reshape_layout(std::string fn, unsigned nx, unsigned ny = 1, unsigned nz = 1, unsigned nw = 1);

    // Read channel object from HDF5 file
    // - Returns empty channel object if channel ID dies not exist
    // - Conversion to float or double is not done for unstructured data
    template <typename dtype> // Read as float or double
    channel<dtype> hdf5_read_channel(std::string fn, unsigned ix = 0, unsigned iy = 0, unsigned iz = 0, unsigned iw = 0);

    // Read single unstructured dataset from HDF5 file
    // - Returns empty std::any if there is no dataset at this location or under this name
    std::any hdf5_read_dset(std::string fn, std::string par_name,
                            unsigned ix = 0, unsigned iy = 0, unsigned iz = 0, unsigned iw = 0,
                            std::string prefix = "par_");

    // Read names of the unstructured datasets from the HDF file
    // - Returns number of the unstructured datasets at the given location
    unsigned long long hdf5_read_dset_names(std::string fn,                                                     // File name
                                            std::vector<std::string> *par_names,                                // Output
                                            unsigned ix = 0, unsigned iy = 0, unsigned iz = 0, unsigned iw = 0, // Location in file
                                            std::string prefix = "par_");                                       // Default prefix

    // Write single unstructured data field to HDF5 file
    // - Scalar types: string, unsigned, int, long long, unsigned long long, float, double
    // - Armadillo Vectors, Matrices and Cubes with types: unsigned, int, unsigned long long, sword, float, double
    // - Vector types: Armadillo Row and Col types are mapped to 1D storage type
    // - arma::Row will be converted to arma::Col
    // - Parameter name may only contain letters and numbers and the underscore "_"
    // - An additional prefix "par_" will be added to the name before writing to the HDF file
    // - Unsupported data types will cause an error
    void hdf5_write_dset(std::string fn, std::string par_name, const std::any *par_data,
                         unsigned ix = 0, unsigned iy = 0, unsigned iz = 0, unsigned iw = 0,
                         std::string prefix = "par_");

}

#endif
