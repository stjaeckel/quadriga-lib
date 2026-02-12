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

#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# qrt_file_parse
Read metadata from a QRT file

## Description:
- Parses a QRT file and extracts metadata such as the number of snapshots, origins, destinations, and frequencies.
- All output arguments are optional; pass `nullptr` to skip any value you don't need.
- Can also retrieve CIR offsets per destination, human-readable names for origins and destinations, and the file version.

## Declaration:
```
void quadriga_lib::qrt_file_parse(
                const std::string &fn,
                arma::uword *no_cir = nullptr,
                arma::uword *no_orig = nullptr,
                arma::uword *no_dest = nullptr,
                arma::uword *no_freq = nullptr,
                arma::uvec *cir_offset = nullptr,
                std::vector<std::string> *orig_names = nullptr,
                std::vector<std::string> *dest_names = nullptr,
                int *version = nullptr)
```

## Arguments:
- `const std::string &**fn**` (input)<br>
  Path to the QRT file.
- `arma::uword ***no_cir** = nullptr` (optional output)<br>
  Number of channel snapshots per origin point.
- `arma::uword ***no_orig** = nullptr` (optional output)<br>
  Number of origin points (TX).
- `arma::uword ***no_dest** = nullptr` (optional output)<br>
  Number of destinations (RX).
- `arma::uword ***no_freq** = nullptr` (optional output)<br>
  Number of frequency bands.
- `arma::uvec ***cir_offset** = nullptr` (optional output)<br>
  CIR offset for each destination. Size `[no_dest]`.
- `std::vector<std::string> ***orig_names** = nullptr` (optional output)<br>
  Names of the origin points (TXs). Size `[no_orig]`.
- `std::vector<std::string> ***dest_names** = nullptr` (optional output)<br>
  Names of the destination points (RXs). Size `[no_dest]`.
- `int ***version** = nullptr` (optional output)<br>
  QRT file version number.

## Example:
```
arma::uword no_cir, no_orig, no_dest, no_freq;
arma::uvec cir_offset;
std::vector<std::string> orig_names, dest_names;
int version;

quadriga_lib::qrt_file_parse("scene.qrt", &no_cir, &no_orig, &no_dest, &no_freq,
                              &cir_offset, &orig_names, &dest_names, &version);
```
MD!*/

void quadriga_lib::qrt_file_parse(const std::string &fn,
                                  arma::uword *no_cir,
                                  arma::uword *no_orig,
                                  arma::uword *no_dest,
                                  arma::uword *no_freq,
                                  arma::uvec *cir_offset,
                                  std::vector<std::string> *orig_names,
                                  std::vector<std::string> *dest_names,
                                  int *version,
                                  arma::fvec *fGHz,
                                  arma::fmat *cir_pos,
                                  arma::fmat *cir_orientation,
                                  arma::fmat *orig_pos,
                                  arma::fmat *orig_orientation,
                                  std::ifstream *file)
{
    // --- Determine which stream to use and whether we own it ----------------
    std::ifstream local_stream;
    bool own_stream = (file == nullptr);

    std::ifstream &fileR = own_stream ? local_stream : *file;

    if (own_stream)
    {
        fileR.open(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::invalid_argument("Cannot open file.");
    }
    else
    {
        // Rewind the supplied stream to the beginning
        fileR.seekg(0, std::ios::beg);
        if (!fileR.good())
            throw std::invalid_argument("Supplied ifstream is not in a good state.");
    }

    // --- Read and validate the header ---------------------------------------
    const std::string bin_id_prefix = "#QRT-BINv";
    std::string bin_id_file(bin_id_prefix.size(), '\0');
    fileR.read(&bin_id_file[0], (std::streamsize)bin_id_prefix.size());

    if (bin_id_file != bin_id_prefix)
        throw std::invalid_argument("Invalid file format: missing QRT-BIN header");

    // Version number (2 ASCII digits)
    std::string version_str(2, '\0');
    fileR.read(&version_str[0], 2);
    int ver = std::stoi(version_str);

    if (ver != 4 && ver != 5 && ver != 6)
        throw std::invalid_argument("Only QRT versions 4, 5 and 6 are supported");

    // --- Global counters ----------------------------------------------------
    unsigned l_no_orig = 0, l_no_cir = 0, l_no_dest = 0, l_no_freq = 1;

    fileR.read((char *)&l_no_orig, sizeof(unsigned));
    fileR.read((char *)&l_no_cir, sizeof(unsigned));
    fileR.read((char *)&l_no_dest, sizeof(unsigned));

    arma::fvec l_freq;
    if (ver > 4)
    {
        fileR.read((char *)&l_no_freq, sizeof(unsigned));
        l_freq.set_size(l_no_freq);
        fileR.read((char *)l_freq.memptr(), l_no_freq * sizeof(float));
    }

    // --- CIR metadata -------------------------------------------------------
    arma::fmat l_cir_pos;
    arma::fmat l_cir_orientation;
    unsigned char cir_fmt = 0;

    if (l_no_cir != 0)
    {
        fileR.read((char *)&cir_fmt, sizeof(unsigned char));

        l_cir_pos.set_size(l_no_cir, 3);
        fileR.read((char *)l_cir_pos.memptr(), l_no_cir * 3 * sizeof(float));

        l_cir_orientation.zeros(l_no_cir, 3);
        if (cir_fmt > 3) // Bank angle
            fileR.read((char *)l_cir_orientation.colptr(0), l_no_cir * sizeof(float));
        if (cir_fmt == 2 || cir_fmt == 3 || cir_fmt == 6 || cir_fmt == 7) // Tilt angle
            fileR.read((char *)l_cir_orientation.colptr(1), l_no_cir * sizeof(float));
        if (cir_fmt == 1 || cir_fmt == 3 || cir_fmt == 5 || cir_fmt == 7) // Heading angle
            fileR.read((char *)l_cir_orientation.colptr(2), l_no_cir * sizeof(float));
    }

    // --- Destination (RX) metadata ------------------------------------------
    arma::u32_vec l_cir_index;
    std::vector<std::string> l_dest_names;

    if (l_no_dest != 0)
    {
        l_cir_index.set_size(l_no_dest);
        fileR.read((char *)l_cir_index.memptr(), l_no_dest * sizeof(unsigned));

        l_dest_names.resize((size_t)l_no_dest);
        for (unsigned i = 0; i < l_no_dest; ++i)
        {
            unsigned char mt_name_length = 0;
            fileR.read((char *)&mt_name_length, sizeof(unsigned char));
            l_dest_names[i].resize((size_t)mt_name_length);
            fileR.read(&l_dest_names[i][0], (size_t)mt_name_length);
        }
    }
    else
    {
        l_cir_index.zeros(1);
        l_dest_names.resize(1);
        l_dest_names[0] = "RX";
    }

    // --- Origin (TX) metadata -----------------------------------------------
    arma::fmat l_orig_pos_all;
    arma::fmat l_orig_orientation;
    arma::u64_vec l_orig_index;

    if (l_no_orig != 0)
    {
        unsigned char bs_fmt = 0;
        fileR.read((char *)&bs_fmt, sizeof(unsigned char));

        l_orig_pos_all.set_size(l_no_orig, 3);
        fileR.read((char *)l_orig_pos_all.memptr(), l_no_orig * 3 * sizeof(float));

        l_orig_orientation.zeros(l_no_orig, 3);
        if (bs_fmt > 3) // Bank angle
            fileR.read((char *)l_orig_orientation.colptr(0), l_no_orig * sizeof(float));
        if (bs_fmt == 2 || bs_fmt == 3 || bs_fmt == 6 || bs_fmt == 7) // Tilt angle
            fileR.read((char *)l_orig_orientation.colptr(1), l_no_orig * sizeof(float));
        if (bs_fmt == 1 || bs_fmt == 3 || bs_fmt == 5 || bs_fmt == 7) // Heading angle
            fileR.read((char *)l_orig_orientation.colptr(2), l_no_orig * sizeof(float));

        l_orig_index.set_size(l_no_orig);
        fileR.read((char *)l_orig_index.memptr(), l_no_orig * sizeof(unsigned long long));
    }

    // --- Populate output parameters -----------------------------------------
    if (no_cir)
        *no_cir = (arma::uword)l_no_cir;
    if (no_orig)
        *no_orig = (arma::uword)l_no_orig;
    if (no_dest)
        *no_dest = (arma::uword)l_no_dest;
    if (no_freq)
        *no_freq = (arma::uword)l_no_freq;
    if (version)
        *version = ver;

    if (l_cir_index.n_elem == 0 || l_cir_index[0] != 0)
        throw std::invalid_argument("Invalid CIR index in QRT file. Potential file corruption.");

    if (cir_offset)
    {
        cir_offset->set_size(l_no_dest);
        auto po = cir_offset->memptr();
        for (auto &val : l_cir_index)
            *po++ = (arma::uword)val;
    }

    if (dest_names)
        *dest_names = std::move(l_dest_names);

    // Read origin names — requires seeking to each origin's data block
    if (orig_names)
    {
        orig_names->clear();
        orig_names->reserve(l_no_orig);
        for (unsigned i = 0; i < l_no_orig; ++i)
        {
            fileR.seekg((std::streampos)l_orig_index(i));

            unsigned char tx_name_length = 0;
            fileR.read((char *)&tx_name_length, sizeof(unsigned char));

            std::string name((size_t)tx_name_length, '\0');
            fileR.read(&name[0], (size_t)tx_name_length);

            orig_names->push_back(std::move(name));
        }
    }

    if (fGHz)
    {
        if (ver == 4 && l_no_orig != 0)
        {
            // In version 4 the frequency is stored per-origin; read from the first origin
            fileR.seekg((std::streampos)l_orig_index(0));

            unsigned char tx_name_length = 0;
            fileR.read((char *)&tx_name_length, sizeof(unsigned char));
            fileR.seekg((std::streamoff)tx_name_length, std::ios::cur); // skip name

            l_freq.set_size(1);
            fileR.read((char *)l_freq.memptr(), sizeof(float));
        }
        *fGHz = std::move(l_freq);
    }

    if (cir_pos)
        *cir_pos = std::move(l_cir_pos);
    if (cir_orientation)
        *cir_orientation = std::move(l_cir_orientation);
    if (orig_pos)
        *orig_pos = std::move(l_orig_pos_all);
    if (orig_orientation)
        *orig_orientation = std::move(l_orig_orientation);

    // --- Close only if we opened the stream ourselves -----------------------
    if (own_stream && fileR.is_open())
        fileR.close();
}

/*!MD
# qrt_read_cache_init
Initialize a QRT read cache for fast repeated access

## Description:
- Reads all fixed metadata from a QRT file into a `qrt_read_cache` struct.
- Pre-computes byte offsets so that subsequent `qrt_file_read` calls only need to
  perform 2 seeks and 4 reads (the per-CIR path data) instead of re-parsing the header.
- Populate the cache once, then pass it to `qrt_file_read` together with a shared
  `std::ifstream` for maximum performance in tight loops.

## Declaration:
```
quadriga_lib::qrt_read_cache quadriga_lib::qrt_read_cache_init(
                const std::string &fn,
                std::ifstream *file = nullptr)
```

## Arguments:
- `const std::string &**fn**` (input)<br>
  Path to the QRT file.

- `std::ifstream ***file** = nullptr` (optional input)<br>
  Optional pre-opened binary ifstream. If `nullptr`, the file is opened from `fn`
  and closed on return. If provided, the stream is left open.

## Returns:
- `qrt_read_cache **cache**` (output)<br>
  Populated cache structure.

## Example:
```
std::ifstream stream("scene.qrt", std::ios::in | std::ios::binary);
auto cache = quadriga_lib::qrt_read_cache_init("scene.qrt", &stream);

arma::vec center_freq, tx_pos, rx_pos;
arma::mat path_gain;
arma::cube M;

for (arma::uword ic = 0; ic < cache.no_cir; ++ic)
    for (arma::uword io = 0; io < cache.no_orig; ++io)
        quadriga_lib::qrt_file_read<double>("", ic, io, true,
            &center_freq, &tx_pos, nullptr, &rx_pos, nullptr,
            nullptr, nullptr, &path_gain, nullptr, &M,
            nullptr, nullptr, nullptr, nullptr, nullptr, 1,
            nullptr, nullptr, &stream, &cache);
```
MD!*/

quadriga_lib::qrt_read_cache quadriga_lib::qrt_read_cache_init(const std::string &fn,
                                                               std::ifstream *file)
{
    quadriga_lib::qrt_read_cache cache;

    // --- Determine which stream to use and whether we own it ----------------
    std::ifstream local_stream;
    bool own_stream = (file == nullptr);
    std::ifstream &fileR = own_stream ? local_stream : *file;

    if (own_stream)
    {
        fileR.open(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::invalid_argument("Cannot open file.");
    }
    else
    {
        fileR.seekg(0, std::ios::beg);
        if (!fileR.good())
            throw std::invalid_argument("Supplied ifstream is not in a good state.");
    }

    // --- Read and validate the header ---------------------------------------
    const std::string bin_id_prefix = "#QRT-BINv";
    std::string bin_id_file(bin_id_prefix.size(), '\0');
    fileR.read(&bin_id_file[0], (std::streamsize)bin_id_prefix.size());

    if (bin_id_file != bin_id_prefix)
        throw std::invalid_argument("Invalid file format: missing QRT-BIN header");

    std::string version_str(2, '\0');
    fileR.read(&version_str[0], 2);
    cache.version = std::stoi(version_str);

    if (cache.version != 4 && cache.version != 5 && cache.version != 6)
        throw std::invalid_argument("Only QRT versions 4, 5 and 6 are supported");

    // --- Global counters ----------------------------------------------------
    fileR.read((char *)&cache.no_orig, sizeof(unsigned));
    fileR.read((char *)&cache.no_cir, sizeof(unsigned));
    fileR.read((char *)&cache.no_dest, sizeof(unsigned));

    cache.no_freq = 1;
    if (cache.version > 4)
    {
        fileR.read((char *)&cache.no_freq, sizeof(unsigned));
        cache.freq.set_size(cache.no_freq);
        fileR.read((char *)cache.freq.memptr(), cache.no_freq * sizeof(float));
    }

    // --- CIR metadata (read all) --------------------------------------------
    if (cache.no_cir != 0)
    {
        unsigned char cir_fmt = 0;
        fileR.read((char *)&cir_fmt, sizeof(unsigned char));

        cache.cir_pos.set_size(cache.no_cir, 3);
        fileR.read((char *)cache.cir_pos.memptr(), cache.no_cir * 3 * sizeof(float));

        cache.cir_orientation.zeros(cache.no_cir, 3);
        if (cir_fmt > 3)
            fileR.read((char *)cache.cir_orientation.colptr(0), cache.no_cir * sizeof(float));
        if (cir_fmt == 2 || cir_fmt == 3 || cir_fmt == 6 || cir_fmt == 7)
            fileR.read((char *)cache.cir_orientation.colptr(1), cache.no_cir * sizeof(float));
        if (cir_fmt == 1 || cir_fmt == 3 || cir_fmt == 5 || cir_fmt == 7)
            fileR.read((char *)cache.cir_orientation.colptr(2), cache.no_cir * sizeof(float));
    }

    // --- Skip destination metadata ------------------------------------------
    if (cache.no_dest != 0)
    {
        fileR.seekg((std::streamoff)(cache.no_dest * sizeof(unsigned)), std::ios::cur);
        for (unsigned i = 0; i < cache.no_dest; ++i)
        {
            unsigned char mt_name_length = 0;
            fileR.read((char *)&mt_name_length, sizeof(unsigned char));
            if (mt_name_length != 0)
                fileR.seekg((std::streamoff)mt_name_length, std::ios::cur);
        }
    }

    // --- Origin metadata (read all) -----------------------------------------
    if (cache.no_orig != 0)
    {
        unsigned char bs_fmt = 0;
        fileR.read((char *)&bs_fmt, sizeof(unsigned char));

        cache.orig_pos_all.set_size(cache.no_orig, 3);
        fileR.read((char *)cache.orig_pos_all.memptr(), cache.no_orig * 3 * sizeof(float));

        cache.orig_orientation.zeros(cache.no_orig, 3);
        if (bs_fmt > 3)
            fileR.read((char *)cache.orig_orientation.colptr(0), cache.no_orig * sizeof(float));
        if (bs_fmt == 2 || bs_fmt == 3 || bs_fmt == 6 || bs_fmt == 7)
            fileR.read((char *)cache.orig_orientation.colptr(1), cache.no_orig * sizeof(float));
        if (bs_fmt == 1 || bs_fmt == 3 || bs_fmt == 5 || bs_fmt == 7)
            fileR.read((char *)cache.orig_orientation.colptr(2), cache.no_orig * sizeof(float));

        cache.orig_index.set_size(cache.no_orig);
        fileR.read((char *)cache.orig_index.memptr(), cache.no_orig * sizeof(unsigned long long));
    }

    // --- Version 4: frequency stored per-origin, read from first origin -----
    if (cache.version == 4 && cache.no_orig != 0)
    {
        fileR.seekg((std::streampos)cache.orig_index(0));
        unsigned char tx_name_length = 0;
        fileR.read((char *)&tx_name_length, sizeof(unsigned char));
        fileR.seekg((std::streamoff)tx_name_length, std::ios::cur);
        cache.freq.set_size(1);
        fileR.read((char *)cache.freq.memptr(), sizeof(float));
    }

    // --- Pre-compute path_data_offset for each origin -----------------------
    // For each origin, this is the absolute byte offset to the start of the
    // path_data_index[] array within that origin's data block.
    // Layout: [tx_name_length(1)] [tx_name(N)] [freq(4, v4 only)] [max_no_path(4)] [path_data_index[no_cir]]
    // So: path_data_offset = orig_index + 1 + tx_name_length + (v4 ? 4 : 0) + 4
    if (cache.no_orig != 0)
    {
        cache.path_data_offset.set_size(cache.no_orig);
        unsigned extra = (cache.version == 4) ? (unsigned)sizeof(float) : 0u;

        for (unsigned i = 0; i < cache.no_orig; ++i)
        {
            fileR.seekg((std::streampos)cache.orig_index(i));
            unsigned char tx_name_length = 0;
            fileR.read((char *)&tx_name_length, sizeof(unsigned char));

            cache.path_data_offset(i) = cache.orig_index(i) + 1ull              // tx_name_length byte
                                        + (unsigned long long)tx_name_length    // tx_name
                                        + (unsigned long long)extra             // freq (v4 only)
                                        + (unsigned long long)sizeof(unsigned); // max_no_path
        }
    }

    // --- Close only if we opened the stream ourselves -----------------------
    if (own_stream && fileR.is_open())
        fileR.close();

    return cache;
}

/*!MD
# qrt_file_read
Read ray-tracing data from a QRT file

## Description:
- Reads channel impulse response (CIR) data from a QRT file for a specific snapshot and origin point.
- Supports both uplink and downlink directions by swapping TX/RX roles accordingly.
- All output arguments are optional; pass `nullptr` to skip any value you don't need.
- The `normalize_M` parameter controls how the polarization transfer matrix `M` and path gains are returned.
- For maximum performance in tight loops, pass a pre-opened `std::ifstream` and a pre-populated
  `qrt_read_cache`. With both, the per-call I/O is reduced to 2 seeks and 4 reads.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
template <typename dtype>
void quadriga_lib::qrt_file_read(
                const std::string &fn,
                arma::uword i_cir = 0,
                arma::uword i_orig = 0,
                bool downlink = true,
                arma::Col<dtype> *center_frequency = nullptr,
                arma::Col<dtype> *tx_pos = nullptr,
                arma::Col<dtype> *tx_orientation = nullptr,
                arma::Col<dtype> *rx_pos = nullptr,
                arma::Col<dtype> *rx_orientation = nullptr,
                arma::Mat<dtype> *fbs_pos = nullptr,
                arma::Mat<dtype> *lbs_pos = nullptr,
                arma::Mat<dtype> *path_gain = nullptr,
                arma::Col<dtype> *path_length = nullptr,
                arma::Cube<dtype> *M = nullptr,
                arma::Col<dtype> *aod = nullptr,
                arma::Col<dtype> *eod = nullptr,
                arma::Col<dtype> *aoa = nullptr,
                arma::Col<dtype> *eoa = nullptr,
                std::vector<arma::Mat<dtype>> *path_coord = nullptr,
                int normalize_M = 1,
                arma::u32_vec *no_int = nullptr,
                arma::fmat *coord = nullptr,
                std::ifstream *file = nullptr,
                const qrt_read_cache *cache = nullptr)
```

## Arguments:
- `const std::string &**fn**` (input)<br>
  Path to the QRT file. Ignored when both `file` and `cache` are provided.

- `arma::uword **i_cir** = 0` (input)<br>
  Snapshot index (0-based).

- `arma::uword **i_orig** = 0` (input)<br>
  Origin index, 0-based. For downlink, origin corresponds to the transmitter.

- `bool **downlink** = true` (input)<br>
  If `true`, origin is TX and destination is RX (downlink). If `false`, roles are swapped (uplink).

- `arma::Col<dtype> ***center_frequency** = nullptr` (optional output)<br>
  Center frequency in Hz. Size `[n_freq]`.

- `arma::Col<dtype> ***tx_pos** = nullptr` (optional output)<br>
  Transmitter position in Cartesian coordinates. Size `[3]`.

- `arma::Col<dtype> ***tx_orientation** = nullptr` (optional output)<br>
  Transmitter orientation (bank, tilt, heading) in radians. Size `[3]`.

- `arma::Col<dtype> ***rx_pos** = nullptr` (optional output)<br>
  Receiver position in Cartesian coordinates. Size `[3]`.

- `arma::Col<dtype> ***rx_orientation** = nullptr` (optional output)<br>
  Receiver orientation (bank, tilt, heading) in radians. Size `[3]`.

- `arma::Mat<dtype> ***fbs_pos** = nullptr` (optional output)<br>
  First-bounce scatterer positions. Size `[3, n_path]`.

- `arma::Mat<dtype> ***lbs_pos** = nullptr` (optional output)<br>
  Last-bounce scatterer positions. Size `[3, n_path]`.

- `arma::Mat<dtype> ***path_gain** = nullptr` (optional output)<br>
  Path gain on linear scale. Size `[n_path, n_freq]`.

- `arma::Col<dtype> ***path_length** = nullptr` (optional output)<br>
  Absolute path length from TX to RX phase center. Size `[n_path]`.

- `arma::Cube<dtype> ***M** = nullptr` (optional output)<br>
  Polarization transfer matrix. Size `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files.

- `arma::Col<dtype> ***aod** = nullptr` (optional output)<br>
  Departure azimuth angles in radians. Size `[n_path]`.

- `arma::Col<dtype> ***eod** = nullptr` (optional output)<br>
  Departure elevation angles in radians. Size `[n_path]`.

- `arma::Col<dtype> ***aoa** = nullptr` (optional output)<br>
  Arrival azimuth angles in radians. Size `[n_path]`.

- `arma::Col<dtype> ***eoa** = nullptr` (optional output)<br>
  Arrival elevation angles in radians. Size `[n_path]`.

- `std::vector<arma::Mat<dtype>> ***path_coord** = nullptr` (optional output)<br>
  Interaction coordinates per path. Vector of length `n_path`, each matrix of size `[3, n_interact + 2]`.

- `int **normalize_M** = 1` (input)<br>
  Normalization option for the polarization transfer matrix.
   0 | `M` as stored in QRT file, `path_gain` is -FSPL
   1 | `M` has sum-column power of 2, `path_gain` is -FSPL minus material losses

- `arma::u32_vec ***no_int** = nullptr` (optional output)<br>
  Number of mesh interactions per path. Size `[n_path]`. A value of 0 indicates a LOS path.

- `arma::fmat ***coord** = nullptr` (optional output)<br>
  Interaction coordinates. Size `[3, sum(no_int)]`.

- `std::ifstream ***file** = nullptr` (optional input)<br>
  Optional pre-opened binary ifstream. If `nullptr`, the file is opened from `fn` and closed
  on return. If provided, the stream is left open after return.

- `const qrt_read_cache ***cache** = nullptr` (optional input)<br>
  Optional pre-parsed metadata cache (from `qrt_read_cache_init`). When provided together with
  `file`, per-call I/O is reduced to 2 seeks and 4 reads.

## Example:
```
arma::vec center_freq, tx_pos, rx_pos, path_length, aod, eod, aoa, eoa;
arma::mat fbs_pos, lbs_pos, path_gain;
arma::cube M;

// Single call (opens and closes file internally):
quadriga_lib::qrt_file_read<double>("scene.qrt", 0, 0, true,
    &center_freq, &tx_pos, nullptr, &rx_pos, nullptr,
    &fbs_pos, &lbs_pos, &path_gain, &path_length, &M,
    &aod, &eod, &aoa, &eoa, nullptr, 1);

// Maximum performance with cache and shared stream:
std::ifstream stream("scene.qrt", std::ios::in | std::ios::binary);
quadriga_lib::qrt_read_cache cache;
quadriga_lib::qrt_read_cache_init("scene.qrt", &cache, &stream);

for (arma::uword ic = 0; ic < cache.no_cir; ++ic)
    for (arma::uword io = 0; io < cache.no_orig; ++io)
        quadriga_lib::qrt_file_read<double>("", ic, io, true,
            &center_freq, &tx_pos, nullptr, &rx_pos, nullptr,
            &fbs_pos, &lbs_pos, &path_gain, &path_length, &M,
            &aod, &eod, &aoa, &eoa, nullptr, 1,
            nullptr, nullptr, &stream, &cache);
```
MD!*/

template <typename dtype>
void quadriga_lib::qrt_file_read(const std::string &fn, arma::uword i_cir, arma::uword i_orig, bool downlink,
                                 arma::Col<dtype> *center_frequency, arma::Col<dtype> *tx_pos, arma::Col<dtype> *tx_orientation,
                                 arma::Col<dtype> *rx_pos, arma::Col<dtype> *rx_orientation,
                                 arma::Mat<dtype> *fbs_pos, arma::Mat<dtype> *lbs_pos,
                                 arma::Mat<dtype> *path_gain, arma::Col<dtype> *path_length, arma::Cube<dtype> *M,
                                 arma::Col<dtype> *aod, arma::Col<dtype> *eod, arma::Col<dtype> *aoa, arma::Col<dtype> *eoa,
                                 std::vector<arma::Mat<dtype>> *path_coord, int normalize_M, arma::u32_vec *no_int, arma::fmat *coord,
                                 std::ifstream *file, const qrt_read_cache *cache)
{
    // === Stream setup =======================================================
    std::ifstream local_stream;
    bool own_stream = (file == nullptr);
    std::ifstream &fileR = own_stream ? local_stream : *file;

    if (own_stream)
    {
        fileR.open(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::invalid_argument("Cannot open file.");
    }

    // === Obtain metadata ====================================================
    int ver;
    unsigned l_no_cir, l_no_freq;
    const float *p_freq;                                  // points at freq data (no copy needed)
    float cir_px, cir_py, cir_pz;                         // CIR position
    float cir_ox = 0.0f, cir_oy = 0.0f, cir_oz = 0.0f;    // CIR orientation
    float orig_px, orig_py, orig_pz;                      // Origin position
    float orig_ox = 0.0f, orig_oy = 0.0f, orig_oz = 0.0f; // Origin orientation
    unsigned long long l_path_data_offset;                // Absolute byte offset to path_data_index array

    arma::fvec l_freq_buf; // Local buffer for slow path; must outlive p_freq usage

    if (cache) // === FAST PATH: metadata from cache (zero I/O) ===============
    {
        ver = cache->version;
        l_no_cir = cache->no_cir;
        l_no_freq = cache->no_freq;
        p_freq = cache->freq.memptr();

        if ((unsigned)i_cir >= l_no_cir)
            throw std::invalid_argument("CIR index exceeds number of CIRs in file.");
        if ((unsigned)i_orig >= cache->no_orig)
            throw std::invalid_argument("Origin (TX) index exceeds number of origin points in file.");

        // CIR position / orientation (column-major: [no_cir, 3])
        cir_px = cache->cir_pos((arma::uword)i_cir, 0);
        cir_py = cache->cir_pos((arma::uword)i_cir, 1);
        cir_pz = cache->cir_pos((arma::uword)i_cir, 2);
        cir_ox = cache->cir_orientation((arma::uword)i_cir, 0);
        cir_oy = cache->cir_orientation((arma::uword)i_cir, 1);
        cir_oz = cache->cir_orientation((arma::uword)i_cir, 2);

        // Origin position / orientation
        orig_px = cache->orig_pos_all((arma::uword)i_orig, 0);
        orig_py = cache->orig_pos_all((arma::uword)i_orig, 1);
        orig_pz = cache->orig_pos_all((arma::uword)i_orig, 2);
        orig_ox = cache->orig_orientation((arma::uword)i_orig, 0);
        orig_oy = cache->orig_orientation((arma::uword)i_orig, 1);
        orig_oz = cache->orig_orientation((arma::uword)i_orig, 2);

        l_path_data_offset = cache->path_data_offset((arma::uword)i_orig);
    }
    else // === SLOW PATH: parse header + selective metadata from file =========
    {
        fileR.seekg(0, std::ios::beg);
        if (!fileR.good())
            throw std::invalid_argument("Supplied ifstream is not in a good state.");

        // --- Read and validate the header -----------------------------------
        const std::string bin_id_prefix = "#QRT-BINv";
        std::string bin_id_file(bin_id_prefix.size(), '\0');
        fileR.read(&bin_id_file[0], (std::streamsize)bin_id_prefix.size());

        if (bin_id_file != bin_id_prefix)
            throw std::invalid_argument("Invalid file format: missing QRT-BIN header");

        std::string version_str(2, '\0');
        fileR.read(&version_str[0], 2);
        ver = std::stoi(version_str);

        if (ver != 4 && ver != 5 && ver != 6)
            throw std::invalid_argument("Only QRT versions 4, 5 and 6 are supported");

        // --- Global counters ------------------------------------------------
        unsigned l_no_orig = 0, l_no_dest = 0;
        l_no_freq = 1;
        fileR.read((char *)&l_no_orig, sizeof(unsigned));
        fileR.read((char *)&l_no_cir, sizeof(unsigned));
        fileR.read((char *)&l_no_dest, sizeof(unsigned));

        if (l_no_orig == 0 || l_no_cir == 0)
            throw std::out_of_range("File does not contain any origins or CIRs.");
        if ((unsigned)i_cir >= l_no_cir)
            throw std::invalid_argument("CIR index exceeds number of CIRs in file.");
        if ((unsigned)i_orig >= l_no_orig)
            throw std::invalid_argument("Origin (TX) index exceeds number of origin points in file.");

        // --- Frequencies (v5/v6: stored in header) --------------------------
        if (ver > 4)
        {
            fileR.read((char *)&l_no_freq, sizeof(unsigned));
            l_freq_buf.set_size(l_no_freq);
            fileR.read((char *)l_freq_buf.memptr(), l_no_freq * sizeof(float));
        }

        // --- CIR metadata (selective read for i_cir) ------------------------
        {
            unsigned char cir_fmt = 0;
            fileR.read((char *)&cir_fmt, sizeof(unsigned char));

            std::streamoff skip_before = (std::streamoff)((unsigned)i_cir * (unsigned)sizeof(float));
            std::streamoff skip_after = (std::streamoff)((l_no_cir - (unsigned)i_cir - 1u) * (unsigned)sizeof(float));
            float val;

            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            cir_px = val;
            fileR.seekg(skip_after, std::ios::cur);

            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            cir_py = val;
            fileR.seekg(skip_after, std::ios::cur);

            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            cir_pz = val;
            fileR.seekg(skip_after, std::ios::cur);

            if (cir_fmt > 3) // Bank angle
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                cir_ox = val;
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (cir_fmt == 2 || cir_fmt == 3 || cir_fmt == 6 || cir_fmt == 7) // Tilt angle
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                cir_oy = val;
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (cir_fmt == 1 || cir_fmt == 3 || cir_fmt == 5 || cir_fmt == 7) // Heading angle
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                cir_oz = val;
                fileR.seekg(skip_after, std::ios::cur);
            }
        }

        // --- Skip destination metadata --------------------------------------
        if (l_no_dest != 0)
        {
            fileR.seekg((std::streamoff)(l_no_dest * sizeof(unsigned)), std::ios::cur);
            for (unsigned i = 0; i < l_no_dest; ++i)
            {
                unsigned char mt_name_length = 0;
                fileR.read((char *)&mt_name_length, sizeof(unsigned char));
                if (mt_name_length != 0)
                    fileR.seekg((std::streamoff)mt_name_length, std::ios::cur);
            }
        }

        // --- Origin metadata (selective read for i_orig) --------------------
        {
            unsigned char bs_fmt = 0;
            fileR.read((char *)&bs_fmt, sizeof(unsigned char));

            std::streamoff skip_before = (std::streamoff)((unsigned)i_orig * (unsigned)sizeof(float));
            std::streamoff skip_after = (std::streamoff)((l_no_orig - (unsigned)i_orig - 1u) * (unsigned)sizeof(float));
            float val;

            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            orig_px = val;
            fileR.seekg(skip_after, std::ios::cur);

            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            orig_py = val;
            fileR.seekg(skip_after, std::ios::cur);

            fileR.seekg(skip_before, std::ios::cur);
            fileR.read((char *)&val, sizeof(float));
            orig_pz = val;
            fileR.seekg(skip_after, std::ios::cur);

            if (bs_fmt > 3) // Bank angle
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                orig_ox = val;
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (bs_fmt == 2 || bs_fmt == 3 || bs_fmt == 6 || bs_fmt == 7) // Tilt angle
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                orig_oy = val;
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (bs_fmt == 1 || bs_fmt == 3 || bs_fmt == 5 || bs_fmt == 7) // Heading angle
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                orig_oz = val;
                fileR.seekg(skip_after, std::ios::cur);
            }

            // Read orig_index[i_orig] and compute path_data_offset
            std::streamoff skip64_before = (std::streamoff)((unsigned)i_orig * (unsigned)sizeof(unsigned long long));
            fileR.seekg(skip64_before, std::ios::cur);

            unsigned long long l_orig_index;
            fileR.read((char *)&l_orig_index, sizeof(unsigned long long));

            // Seek to the origin block, read tx_name_length, compute offset
            fileR.seekg((std::streampos)l_orig_index);
            unsigned char tx_name_length = 0;
            fileR.read((char *)&tx_name_length, sizeof(unsigned char));

            l_path_data_offset = l_orig_index + 1ull + (unsigned long long)tx_name_length + (unsigned long long)((ver == 4) ? sizeof(float) : 0u) + (unsigned long long)sizeof(unsigned);
        }

        // --- Version 4: frequency stored per-origin -------------------------
        if (ver == 4)
        {
            // Frequency is at: path_data_offset - sizeof(unsigned) - sizeof(float)
            //                 = orig_index + 1 + tx_name_length
            unsigned long long freq_offset = l_path_data_offset - (unsigned long long)sizeof(unsigned) - (unsigned long long)sizeof(float);
            fileR.seekg((std::streampos)freq_offset);
            l_freq_buf.set_size(1);
            fileR.read((char *)l_freq_buf.memptr(), sizeof(float));
        }

        p_freq = l_freq_buf.memptr();
    }

    arma::uword no_freq = (arma::uword)l_no_freq;
    bool v6 = (ver == 6);

    // === Read CIR path data (2 seeks + 4 reads) =============================
    // Seek directly to path_data_index[i_cir] within the origin block
    fileR.seekg((std::streampos)(l_path_data_offset + (unsigned long long)i_cir * sizeof(unsigned long long)));

    unsigned long long data_offset = 0;
    fileR.read((char *)&data_offset, sizeof(unsigned long long));

    // Seek to the actual CIR data block
    fileR.seekg((std::streampos)data_offset);

    // Number of paths
    unsigned no_path;
    fileR.read((char *)&no_path, sizeof(unsigned));

    // Number of mesh interactions per path
    arma::u32_vec no_intR;
    unsigned sum_no_int = 0;
    no_intR.set_size(no_path);
    {
        unsigned *p_no_int = no_intR.memptr();
        for (unsigned iP = 0; iP < no_path; ++iP)
        {
            unsigned char no_mesh_interact_byte = 0;
            fileR.read((char *)&no_mesh_interact_byte, sizeof(unsigned char));
            p_no_int[iP] = (unsigned)no_mesh_interact_byte;
            sum_no_int += p_no_int[iP];
        }
    }

    // Polarization transfer matrix
    size_t xprmat_size = v6 ? 2 : 8;
    arma::fcube xprmatR(xprmat_size, no_path, l_no_freq);
    fileR.read((char *)xprmatR.memptr(), xprmat_size * no_path * l_no_freq * sizeof(float));

    // Interaction coordinates
    arma::fmat coordR(3, sum_no_int);
    fileR.read((char *)coordR.memptr(), 3 * sum_no_int * sizeof(float));

    // === Close only if we opened the stream ourselves =======================
    if (own_stream && fileR.is_open())
        fileR.close();

    // === Gain at 1 m (FSPL reference) =======================================
    arma::vec gain_at_1m(no_freq);
    for (arma::uword i_freq = 0; i_freq < no_freq; ++i_freq)
        gain_at_1m[i_freq] = -32.45 - 20.0 * std::log10((double)p_freq[i_freq]);

    // === Output: center_frequency ===========================================
    if (center_frequency)
    {
        center_frequency->set_size(no_freq);
        for (arma::uword i_freq = 0; i_freq < no_freq; ++i_freq)
            center_frequency->at(i_freq) = (dtype)p_freq[i_freq] * (dtype)1e9;
    }

    // === Output: positions ==================================================
    dtype Ox = (dtype)orig_px, Oy = (dtype)orig_py, Oz = (dtype)orig_pz;
    dtype Dx = (dtype)cir_px, Dy = (dtype)cir_py, Dz = (dtype)cir_pz;

    if (tx_pos && downlink)
        *tx_pos = {Ox, Oy, Oz};
    if (rx_pos && !downlink)
        *rx_pos = {Ox, Oy, Oz};

    if (rx_pos && downlink)
        *rx_pos = {Dx, Dy, Dz};
    if (tx_pos && !downlink)
        *tx_pos = {Dx, Dy, Dz};

    // === Output: orientations ===============================================
    if (tx_orientation && downlink)
        *tx_orientation = {(dtype)orig_ox, (dtype)orig_oy, (dtype)orig_oz};
    if (tx_orientation && !downlink)
        *tx_orientation = {(dtype)cir_ox, (dtype)cir_oy, (dtype)cir_oz};

    if (rx_orientation && downlink)
        *rx_orientation = {(dtype)cir_ox, (dtype)cir_oy, (dtype)cir_oz};
    if (rx_orientation && !downlink)
        *rx_orientation = {(dtype)orig_ox, (dtype)orig_oy, (dtype)orig_oz};

    // === Output: raw interaction data =======================================
    if (no_int)
        *no_int = no_intR;

    if (coord)
        *coord = coordR;

    // === Calculate path gain and polarization matrix M ======================
    // - xprmatR includes all interaction losses, but not the FSPL
    // - here we calculate the normalized polarization matrix M and the PG without FSPL
    if (M || path_gain)
    {
        dtype *dst = nullptr;
        if (M)
        {
            if (v6)
                M->set_size(2, no_path, no_freq);
            else
                M->set_size(8, no_path, no_freq);
            dst = M->memptr();
        }

        dtype *pg = nullptr;
        if (path_gain)
        {
            path_gain->set_size(no_path, no_freq);
            pg = path_gain->memptr();
        }

        const float *src = xprmatR.memptr();

        for (arma::uword i_freq = 0; i_freq < no_freq; ++i_freq)
            for (arma::uword i_path = 0; i_path < no_path; ++i_path)
            {
                // load as dtype into registers
                const dtype r11 = (dtype)src[0];
                const dtype i11 = (dtype)src[1];
                const dtype r21 = v6 ? 0.0 : (dtype)src[2];
                const dtype i21 = v6 ? 0.0 : (dtype)src[3];
                const dtype r12 = v6 ? 0.0 : (dtype)src[4];
                const dtype i12 = v6 ? 0.0 : (dtype)src[5];
                const dtype r22 = v6 ? 0.0 : (dtype)src[6];
                const dtype i22 = v6 ? 0.0 : (dtype)src[7];

                // column powers (V and H) - path gain = max column power
                dtype gain = 1.0;
                if (normalize_M == 1)
                {
                    const dtype p1 = r11 * r11 + i11 * i11 + r21 * r21 + i21 * i21;
                    const dtype p2 = r12 * r12 + i12 * i12 + r22 * r22 + i22 * i22;
                    gain = (p1 > p2) ? p1 : p2;
                }

                // write path gain
                if (pg)
                    *pg++ = gain;

                // normalization factor: max column power -> 1
                dtype scale = (dtype)0;
                if (gain > (dtype)0)
                    scale = dtype(1.0 / std::sqrt((double)gain));

                if (dst)
                {
                    if (downlink) // copy, normalize
                    {
                        dst[0] = r11 * scale;
                        dst[1] = i11 * scale;
                        if (!v6)
                        {
                            dst[2] = r21 * scale;
                            dst[3] = i21 * scale;
                            dst[4] = r12 * scale;
                            dst[5] = i12 * scale;
                            dst[6] = r22 * scale;
                            dst[7] = i22 * scale;
                        }
                    }
                    else // uplink: conjugate transpose, normalize
                    {
                        // H_UL = H_DL^H
                        dst[0] = r11 * scale;  // Re(h11)
                        dst[1] = -i11 * scale; // -Im(h11)
                        if (!v6)
                        {
                            dst[2] = r12 * scale;  // Re(h12)
                            dst[3] = -i12 * scale; // -Im(h12)
                            dst[4] = r21 * scale;  // Re(h21)
                            dst[5] = -i21 * scale; // -Im(h21)
                            dst[6] = r22 * scale;  // Re(h22)
                            dst[7] = -i22 * scale; // -Im(h22)
                        }
                    }
                    dst += v6 ? 2 : 8;
                }
                src += v6 ? 2 : 8;
            }
    }

    bool want_angles = aod || eod || aoa || eoa;
    bool want_length = path_gain || path_length;

    // === Extract path metadata ==============================================
    // - here we add the FSPL from path length to the PG
    if (want_angles || want_length || fbs_pos || lbs_pos || path_coord)
    {
        // Convert interaction coordinates to desired precision (e.g. float to double)
        arma::Mat<dtype> coordD(3, coordR.n_cols, arma::fill::none);
        {
            dtype *p = coordD.memptr();
            for (auto &val : coordR)
                *p++ = (dtype)val;
        }

        // Convert path interaction coordinates into FBS/LBS positions, path length and angles
        arma::Mat<dtype> path_angles;
        arma::Col<dtype> path_length_local;

        if (want_angles && want_length)
            quadriga_lib::coord2path<dtype>(Ox, Oy, Oz, Dx, Dy, Dz, &no_intR, &coordD,
                                            &path_length_local, fbs_pos, lbs_pos, &path_angles, path_coord, !downlink);
        else if (want_angles && !want_length)
            quadriga_lib::coord2path<dtype>(Ox, Oy, Oz, Dx, Dy, Dz, &no_intR, &coordD,
                                            nullptr, fbs_pos, lbs_pos, &path_angles, path_coord, !downlink);
        else if (!want_angles && want_length)
            quadriga_lib::coord2path<dtype>(Ox, Oy, Oz, Dx, Dy, Dz, &no_intR, &coordD,
                                            &path_length_local, fbs_pos, lbs_pos, nullptr, path_coord, !downlink);
        else // want_none
            quadriga_lib::coord2path<dtype>(Ox, Oy, Oz, Dx, Dy, Dz, &no_intR, &coordD,
                                            nullptr, fbs_pos, lbs_pos, nullptr, path_coord, !downlink);

        // Adjust path gain to include the FSPL
        if (want_length)
        {
            if (path_length)
                path_length->set_size(no_path);

            dtype *src = path_length_local.memptr();
            dtype *pg = path_gain ? path_gain->memptr() : nullptr;
            dtype *pl = path_length ? path_length->memptr() : nullptr;
            double *p_gain_at_1m = gain_at_1m.memptr();

            for (arma::uword i_path = 0; i_path < no_path; ++i_path)
            {
                dtype len = src[i_path];
                if (pl)
                    pl[i_path] = len;
                if (pg)
                {
                    for (arma::uword i_freq = 0; i_freq < no_freq; ++i_freq)
                    {
                        double gainFS = p_gain_at_1m[i_freq] - 20.0 * std::log10((double)len);
                        pg[i_freq * no_path + i_path] *= (dtype)std::pow(10.0, 0.1 * gainFS);
                    }
                }
            }
        }

        if (aod)
        {
            aod->set_size(no_path);
            std::memcpy(aod->memptr(), path_angles.colptr(0), no_path * sizeof(dtype));
        }

        if (eod)
        {
            eod->set_size(no_path);
            std::memcpy(eod->memptr(), path_angles.colptr(1), no_path * sizeof(dtype));
        }

        if (aoa)
        {
            aoa->set_size(no_path);
            std::memcpy(aoa->memptr(), path_angles.colptr(2), no_path * sizeof(dtype));
        }

        if (eoa)
        {
            eoa->set_size(no_path);
            std::memcpy(eoa->memptr(), path_angles.colptr(3), no_path * sizeof(dtype));
        }
    }
}

template void quadriga_lib::qrt_file_read(const std::string &fn, arma::uword i_cir, arma::uword i_orig, bool downlink,
                                          arma::Col<float> *center_frequency, arma::Col<float> *tx_pos, arma::Col<float> *tx_orientation,
                                          arma::Col<float> *rx_pos, arma::Col<float> *rx_orientation,
                                          arma::Mat<float> *fbs_pos, arma::Mat<float> *lbs_pos,
                                          arma::Mat<float> *path_gain, arma::Col<float> *path_length, arma::Cube<float> *M,
                                          arma::Col<float> *aod, arma::Col<float> *eod, arma::Col<float> *aoa, arma::Col<float> *eoa,
                                          std::vector<arma::Mat<float>> *path_coord, int normalize_M, arma::u32_vec *no_int, arma::fmat *coord,
                                          std::ifstream *file, const qrt_read_cache *cache);

template void quadriga_lib::qrt_file_read(const std::string &fn, arma::uword i_cir, arma::uword i_orig, bool downlink,
                                          arma::Col<double> *center_frequency, arma::Col<double> *tx_pos, arma::Col<double> *tx_orientation,
                                          arma::Col<double> *rx_pos, arma::Col<double> *rx_orientation,
                                          arma::Mat<double> *fbs_pos, arma::Mat<double> *lbs_pos,
                                          arma::Mat<double> *path_gain, arma::Col<double> *path_length, arma::Cube<double> *M,
                                          arma::Col<double> *aod, arma::Col<double> *eod, arma::Col<double> *aoa, arma::Col<double> *eoa,
                                          std::vector<arma::Mat<double>> *path_coord, int normalize_M, arma::u32_vec *no_int, arma::fmat *coord,
                                          std::ifstream *file, const qrt_read_cache *cache);
