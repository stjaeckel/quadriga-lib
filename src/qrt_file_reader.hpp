// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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

#ifndef qdlib_qrt_file_reader_H
#define qdlib_qrt_file_reader_H

#include <armadillo>
#include <string>

// Class to iterate through bin files that contain the RT results
class qrt_file_reader
{
public:
    std::ifstream fileR;                 // File input stream
    unsigned no_orig = 0;                // Number of origin (TX) positions in the file, may be 0
    unsigned no_cir = 0;                 // Number of channel impulse responses, same for each TX
    unsigned no_dest = 0;                // Number of destination (RX) positions (groups of CIRs)
    unsigned char cir_fmt = 7;           // CIR orientation format
    arma::fmat cir_pos;                  // CIR positions in Cartesian coordinates, size [no_cir, 3]
    arma::fmat cir_orientation;          // CIR orientation in Euler angles in rad, size [no_cir, 3]
    arma::u32_vec cir_index;             // CIR offset for each RX, 0-based, length [no_dest]
    std::vector<std::string> dest_names; // Destination (RX) names
    arma::fmat orig_pos_all;             // Origin (TX) positions, size [no_orig, 3]
    arma::fmat orig_orientation;         // Origin (TX) orientations, Euler angels in rad, size [no_orig, 3]
    arma::u64_vec orig_index;            // Index of the origin (TX) data, bytes from BOF, vector of length [no_cir]
    std::string orig_name;               // Current origin (TX) name
    float orig_pos[3] = {0, 0, 0};       // Current origin (TX) position
    float fGHz = 0.0f;                   // Carrier frequency in GHz
    unsigned max_no_path = 0;            // Maximum number of paths per RX
    arma::u64_vec path_data_index;       // Index of the TX-RX path data, bytes from BOF, size [no_cir]
    arma::u32_vec no_int;                // Number of mesh interactions per path of current RX, vector of length [no_path], 0=LOS
    arma::fmat xprmat;                   // Polarization transfer matrix, size [8, n_path], interleaved complex
    arma::fmat coord;                    // Interaction coordinates, size [3, sum(no_int)]

    // Default constructor
    qrt_file_reader() {};

    // CONSTRUCTOR: Opens file and reads all metadata
    qrt_file_reader(std::string fn)
    {
        fileR = std::ifstream(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::invalid_argument("Cannot open file.");

        // Read the ID_string and check if it is correct
        const std::string bin_id = "#QRT-BINv04";
        std::string bin_id_file(bin_id.size(), '\0');
        fileR.read(&bin_id_file[0], bin_id.size());

        if (fileR.gcount() != static_cast<std::streamsize>(bin_id.size()) || bin_id_file != bin_id)
            throw std::invalid_argument("Wrong file format.");

        fileR.read((char *)&no_orig, sizeof(unsigned));
        fileR.read((char *)&no_cir, sizeof(unsigned));
        fileR.read((char *)&no_dest, sizeof(unsigned));

        // Read CIR metadata
        if (no_cir != 0)
        {
            fileR.read((char *)&cir_fmt, sizeof(unsigned char));

            cir_pos.set_size(no_cir, 3);
            fileR.read((char *)cir_pos.memptr(), no_cir * 3 * sizeof(float));

            cir_orientation.zeros(no_cir, 3);
            if (cir_fmt > 3) // Bank angle
                fileR.read((char *)cir_orientation.colptr(0), no_cir * sizeof(float));
            if (cir_fmt == 2 || cir_fmt == 3 || cir_fmt == 6 || cir_fmt == 7) // Tilt angle
                fileR.read((char *)cir_orientation.colptr(1), no_cir * sizeof(float));
            if (cir_fmt == 1 || cir_fmt == 3 || cir_fmt == 5 || cir_fmt == 7) // Heading angle
                fileR.read((char *)cir_orientation.colptr(2), no_cir * sizeof(float));
        }
        else
            cir_fmt = 0;

        // Read MT metadata
        if (no_dest != 0)
        {
            cir_index.set_size(no_dest);
            fileR.read((char *)cir_index.memptr(), no_dest * sizeof(unsigned));

            dest_names.resize((size_t)no_dest);
            for (unsigned i = 0; i < no_dest; ++i)
            {
                unsigned char mt_name_length = 0;
                fileR.read((char *)&mt_name_length, sizeof(unsigned char));

                dest_names[i].resize((size_t)mt_name_length);
                fileR.read(&dest_names[i][0], (size_t)mt_name_length);
            }
        }
        else
        {
            cir_index.zeros(1);
            dest_names.resize(1);
            dest_names[1] = "RX";
        }

        // Read BS metadata
        if (no_orig != 0)
        {
            unsigned char bs_fmt = 0;
            fileR.read((char *)&bs_fmt, sizeof(unsigned char));

            orig_pos_all.set_size(no_orig, 3);
            fileR.read((char *)orig_pos_all.memptr(), no_orig * 3 * sizeof(float));

            orig_orientation.zeros(no_orig, 3);
            if (bs_fmt > 3) // Bank angle
                fileR.read((char *)orig_orientation.colptr(0), no_orig * sizeof(float));
            if (bs_fmt == 2 || bs_fmt == 3 || bs_fmt == 6 || bs_fmt == 7) // Tilt angle
                fileR.read((char *)orig_orientation.colptr(1), no_orig * sizeof(float));
            if (bs_fmt == 1 || bs_fmt == 3 || bs_fmt == 5 || bs_fmt == 7) // Heading angle
                fileR.read((char *)orig_orientation.colptr(2), no_orig * sizeof(float));

            orig_index.set_size(no_orig);
            fileR.read((char *)orig_index.memptr(), no_orig * sizeof(unsigned long long));

            set_orig(0);
        }
    }

    // CONSTRUCTOR: Opens file and reads ONLY metadata for the selected iCIR and iORIG
    // - uses minimum amount of read operations to get needed data
    // - reads no_orig, no_cir, no_dest, cir_fmt
    // - Initializes members: cir_pos, cir_orientation, orig_pos_all, orig_orientation, orig_index
    // - These member will only contain a single entry!
    // - For the selected iCIR, sets: cir_pos(iCIR,:) and cir_orientation(iCIR,:) to values frm the file
    // - For the selected iORIG, sets: orig_pos_all(iORIG,:) and orig_orientation(iORIG,:) to values frm the file
    // - cir_index and dest_names are not initializes (empty)
    qrt_file_reader(std::string fn, unsigned iCIR, unsigned iORIG)
    {
        fileR = std::ifstream(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::invalid_argument("Cannot open file.");

        // Read the ID_string and check if it is correct
        const std::string bin_id = "#QRT-BINv04";
        std::string bin_id_file(bin_id.size(), '\0');
        fileR.read(&bin_id_file[0], bin_id.size());

        if (fileR.gcount() != (std::streamsize)bin_id.size() || bin_id_file != bin_id)
            throw std::invalid_argument("Wrong file format.");

        // Global counters
        fileR.read((char *)&no_orig, sizeof(unsigned));
        fileR.read((char *)&no_cir, sizeof(unsigned));
        fileR.read((char *)&no_dest, sizeof(unsigned));

        if (no_orig == 0 || no_cir == 0)
            throw std::out_of_range("File does not contain any origins or CIRs.");

        if (iCIR >= no_cir || iORIG >= no_orig)
            throw std::out_of_range("Requested iCIR or iORIG out of range.");

        // CIR metadata: read cir_fmt, allocate/zero matrices, read only requested CIR position + orientation
        if (no_cir != 0)
        {
            fileR.read((char *)&cir_fmt, sizeof(unsigned char));

            cir_pos.set_size(1, 3);
            cir_orientation.zeros(1, 3);
            float *p_pos = cir_pos.memptr(), *p_ori = cir_orientation.memptr();

            float val = 0.0f;
            std::streamoff skip_before = (std::streamoff)(iCIR * (unsigned)sizeof(float));
            std::streamoff skip_after = (std::streamoff)((no_cir - iCIR - 1u) * (unsigned)sizeof(float));

            // cir_pos_x[no_cir]
            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            p_pos[0] = val;
            fileR.seekg(skip_after, std::ios::cur);

            // cir_pos_y[no_cir]
            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            p_pos[1] = val;
            fileR.seekg(skip_after, std::ios::cur);

            // cir_pos_z[no_cir]
            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            p_pos[2] = val;
            fileR.seekg(skip_after, std::ios::cur);

            // Orientation arrays (all length no_cir)
            if (cir_fmt > 3)
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                p_ori[0] = val; // bank
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (cir_fmt == 2 || cir_fmt == 3 || cir_fmt == 6 || cir_fmt == 7)
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                p_ori[1] = val; // tilt
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (cir_fmt == 1 || cir_fmt == 3 || cir_fmt == 5 || cir_fmt == 7)
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                p_ori[2] = val; // heading
                fileR.seekg(skip_after, std::ios::cur);
            }
        }
        else
            cir_fmt = 0;

        // MT metadata (no_dest): skip entirely (do not fill cir_index or dest_names)
        if (no_dest != 0)
        {
            // Skip mt_cir_index[no_dest]
            std::streamoff skip = (std::streamoff)(no_dest * (unsigned)sizeof(unsigned));
            fileR.seekg(skip, std::ios::cur);

            // Skip mt_names
            for (unsigned i = 0; i < no_dest; ++i)
            {
                unsigned char mt_name_length = 0;
                fileR.read((char *)&mt_name_length, sizeof(unsigned char));
                if (mt_name_length != 0)
                    fileR.seekg((std::streamoff)mt_name_length, std::ios::cur);
            }
        }

        // BS metadata (origins): allocate/zero matrices and orig_index, read only the selected origin position + orientation + index
        if (no_orig != 0)
        {
            unsigned char bs_fmt = 0;
            fileR.read((char *)&bs_fmt, sizeof(unsigned char));

            orig_pos_all.set_size(1, 3);
            orig_orientation.zeros(1, 3);
            orig_index.set_size(1);
            float *p_pos = orig_pos_all.memptr(), *p_ori = orig_orientation.memptr();

            float val = 0.0f;
            std::streamoff skip_before = (std::streamoff)(iORIG * (unsigned)sizeof(float));
            std::streamoff skip_after = (std::streamoff)((no_orig - iORIG - 1u) * (unsigned)sizeof(float));

            // bs_pos_x[no_orig]
            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            p_pos[0] = val;
            fileR.seekg(skip_after, std::ios::cur);

            // bs_pos_y[no_orig]
            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            p_pos[1] = val;
            fileR.seekg(skip_after, std::ios::cur);

            // bs_pos_z[no_orig]
            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            p_pos[2] = val;
            fileR.seekg(skip_after, std::ios::cur);

            // BS orientation arrays (all length no_orig)
            if (bs_fmt > 3)
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                p_ori[0] = val; // bank
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (bs_fmt == 2 || bs_fmt == 3 || bs_fmt == 6 || bs_fmt == 7)
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                p_ori[1] = val; // tilt
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (bs_fmt == 1 || bs_fmt == 3 || bs_fmt == 5 || bs_fmt == 7)
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                p_ori[2] = val; // heading
                fileR.seekg(skip_after, std::ios::cur);
            }

            // bs_data_index[no_orig] -> orig_index
            std::streamoff skip64_before = (std::streamoff)(iORIG * (unsigned)sizeof(unsigned long long));
            std::streamoff skip64_after = (std::streamoff)((no_orig - iORIG - 1u) * (unsigned)sizeof(unsigned long long));

            fileR.seekg(skip64_before, std::ios::cur);
            unsigned long long idx = 0;
            fileR.read((char *)&idx, sizeof(unsigned long long));
            orig_index[0] = idx;
            fileR.seekg(skip64_after, std::ios::cur);

            set_orig(0);
        }
    }

    // Set TX and read TX metadata
    void set_orig(unsigned iORIG)
    {
        if (iORIG >= no_orig)
            throw std::invalid_argument("BS index exceeds number of BSs in file.");

        fileR.seekg(orig_index(iORIG));

        unsigned char tx_name_length = 0;
        fileR.read((char *)&tx_name_length, sizeof(unsigned char));

        orig_name.resize((size_t)tx_name_length);
        fileR.read(&orig_name[0], (size_t)tx_name_length);

        orig_pos[0] = orig_pos_all(iORIG, 0);
        orig_pos[1] = orig_pos_all(iORIG, 1);
        orig_pos[2] = orig_pos_all(iORIG, 2);

        fileR.read((char *)&fGHz, sizeof(float));
        fileR.read((char *)&max_no_path, sizeof(unsigned));

        path_data_index.set_size(no_cir);
        fileR.read((char *)path_data_index.memptr(), no_cir * sizeof(unsigned long long));
    }

    // Random access to a specific CIR
    unsigned read_cir(unsigned iORIG, unsigned iCIR, arma::u32_vec &no_intR, arma::fmat &xprmatR, arma::fmat &coordR)
    {
        fileR.seekg(orig_index(iORIG), std::ios::beg);

        // Read name length
        unsigned char tx_name_length = 0;
        fileR.read((char *)&tx_name_length, sizeof(tx_name_length));

        // Skip over the name bytes + fGHz + max_no_path + skip to the single element in path_data_index[iCIR]
        long long skip = tx_name_length + sizeof(float) + sizeof(unsigned) + iCIR * sizeof(unsigned long long);
        fileR.seekg(skip, std::ios::cur);

        unsigned long long data_offset = 0;
        fileR.read((char *)&data_offset, sizeof(unsigned long long));

        // Seek to the data block
        fileR.seekg(data_offset, std::ios::beg);

        // Number of paths
        unsigned no_path;
        fileR.read((char *)&no_path, sizeof(unsigned));

        // Read number of mesh interactions from file
        unsigned sum_no_int = 0;
        no_intR.set_size(no_path);
        unsigned *p_no_int = no_intR.memptr();
        for (unsigned iP = 0; iP < no_path; ++iP)
        {
            unsigned char no_mesh_interact_byte = 0;
            fileR.read((char *)&no_mesh_interact_byte, sizeof(unsigned char));
            p_no_int[iP] = (unsigned)no_mesh_interact_byte;
            sum_no_int += p_no_int[iP];
        }

        // Read polarization transfer matrix
        xprmatR.set_size(8, no_path);
        fileR.read((char *)xprmatR.memptr(), 8 * no_path * sizeof(float));

        // Read interaction coordinates
        coordR.set_size(3, sum_no_int);
        fileR.read((char *)coordR.memptr(), 3 * sum_no_int * sizeof(float));

        return no_path;
    }

    // Closes file and clear memory
    void close()
    {
        no_orig = 0;
        no_cir = 0;
        no_dest = 0;

        cir_pos.reset();
        cir_orientation.reset();
        cir_index.reset();
        dest_names.clear();
        dest_names.shrink_to_fit();

        orig_pos_all.reset();
        orig_orientation.reset();
        orig_index.reset();
        orig_name.clear();

        orig_pos[0] = 0.0f;
        orig_pos[1] = 0.0f;
        orig_pos[2] = 0.0f;
        fGHz = 0.0f;
        max_no_path = 0;

        path_data_index.reset();
        no_int.reset();
        xprmat.reset();
        coord.reset();

        if (fileR.is_open())
            fileR.close();
    }
};

#endif