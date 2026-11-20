// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"
#include <cstring>
#include <limits>

static_assert(sizeof(arma::uword) == sizeof(unsigned long long), "arma::uword and unsigned long long have different sizes");
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t and unsigned long long have different sizes");

// Checks of the QRT readers against the file: every count is compared with the bytes that remain in the file
// before memory is allocated for it, and the stream state is checked after the reads.
namespace
{
    // Size of the file behind the stream in bytes; moves the read position, the caller seeks afterwards
    inline size_t qrt_file_size(std::ifstream &f)
    {
        std::streamoff n = (std::streamoff)f.rdbuf()->pubseekoff(0, std::ios::end, std::ios::in);
        if (n < 0)
            throw std::runtime_error("Cannot determine the size of the QRT file.");
        return (size_t)n;
    }

    inline void qrt_check_stream(const std::ifstream &f)
    {
        if (!f)
            throw std::runtime_error("QRT file truncated or corrupt.");
    }

    // Throws if 'count' items of 'item_bytes' bytes each do not fit into the file from byte 'pos' on
    inline void qrt_check_fits(size_t file_size, size_t pos, size_t count, size_t item_bytes, const char *what)
    {
        size_t remaining = pos < file_size ? file_size - pos : 0;
        if (count != 0 && count > remaining / item_bytes)
            throw std::runtime_error(std::string("QRT file truncated or corrupt: ") + what + " = " +
                                     std::to_string(count) + " does not fit into the file.");
    }

    // Same, after checking that the reads so far succeeded; the readers track 'pos' themselves (tellg costs a syscall)
    inline void qrt_check_fits(const std::ifstream &f, size_t file_size, size_t pos, size_t count, size_t item_bytes, const char *what)
    {
        qrt_check_stream(f);
        qrt_check_fits(file_size, pos, count, item_bytes, what);
    }

    // Throws if a stored byte offset does not point into the file
    inline void qrt_check_offset(size_t file_size, size_t offset, const char *what)
    {
        if (offset >= file_size)
            throw std::runtime_error(std::string("QRT file truncated or corrupt: ") + what + " points past the end of the file.");
    }

    // Reads the ID "#QRT-BINv" and the two-digit version from the start of the file; returns the version (4, 5 or 6)
    inline int qrt_read_header(std::ifstream &f)
    {
        char id[11];
        f.read(id, 11);
        if (f.gcount() < 9 || std::memcmp(id, "#QRT-BINv", 9) != 0)
            throw std::runtime_error("Invalid file format: missing QRT-BIN header.");
        qrt_check_stream(f);
        int ver = (id[9] >= '0' && id[9] <= '9' && id[10] >= '0' && id[10] <= '9') ? 10 * (id[9] - '0') + (id[10] - '0') : 0;
        if (ver != 4 && ver != 5 && ver != 6)
            throw std::runtime_error("Only QRT versions 4, 5 and 6 are supported.");
        return ver;
    }

    // Reads the frequency count and list of a v5 / v6 header (they follow the three counters); advances 'pos'
    inline void qrt_read_freq_list(std::ifstream &f, size_t file_size, size_t &pos, unsigned &n_freq, arma::fvec &freq)
    {
        f.read((char *)&n_freq, sizeof(unsigned));
        pos += sizeof(unsigned);
        qrt_check_fits(f, file_size, pos, n_freq, sizeof(float), "n_freq");
        freq.set_size(n_freq);
        f.read((char *)freq.memptr(), n_freq * sizeof(float));
        pos += (size_t)n_freq * sizeof(float);
    }

    // Number of stored orientation columns (bank, tilt, head) for a format byte, as the readers decode it
    inline size_t qrt_orientation_columns(uint8_t fmt)
    {
        size_t n = (fmt > 3) ? 1 : 0;
        if (fmt == 2 || fmt == 3 || fmt == 6 || fmt == 7)
            ++n;
        if (fmt == 1 || fmt == 3 || fmt == 5 || fmt == 7)
            ++n;
        return n;
    }
}

// Factor from the frequencies stored in a QRT file of the given version to Hz: 1 or 1e9.
// v4 files store GHz and v6 files Hz. v5 files written before the library stored Hz hold GHz, so a v5 list
// whose values are all below 1000 is taken as GHz (an EM frequency below 1 kHz does not occur).
// 'freq' holds the 'n_freq' stored values.
static inline double qd_qrt_freq_scale(const float *freq, size_t n_freq, int version)
{
    if (version == 6)
        return 1.0;
    if (version == 5)
        for (size_t i = 0; i < n_freq; ++i)
            if (freq[i] >= 1000.0f)
                return 1.0;
    return 1.0e9;
}

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# qrt_read_cache_init
Initialize a QRT read cache for fast repeated access

- Reads all fixed metadata from a QRT file into a `quadriga_lib::qrt_read_cache` struct.
- Pre-computes byte offsets so that [[qrt_file_read]] needs only 2 seeks and 4 reads per CIR instead of re-parsing the header.
- Populate once, then pass the cache and a shared `std::ifstream` to [[qrt_file_read]] calls that read a file in parts.
- If `file` is `nullptr`, the file is opened internally and closed on return; if provided, the stream is left open.
- For a TX slot reserved by [[qrt_file_init]] but not yet written, `orig_index` and `path_data_offset`
  are `0`; [[qrt_file_read]] treats that as an empty (unwritten) origin.
- Throws `std::runtime_error` if the file cannot be opened, is not a QRT file of a supported version, is truncated,
  or if a count or offset in it does not fit the file size.

## Declaration:
```
quadriga_lib::qrt_read_cache quadriga_lib::qrt_read_cache_init(
    const std::string &fn,
    std::ifstream *file = nullptr);
```

## Inputs:
- **`fn`** — Path to the QRT file
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally

## Returns:
- Populated `quadriga_lib::qrt_read_cache` struct with the following members:<br><br>
  | Member             | Type         | Description                                                                                     |
  | ------------------ | ------------ | ----------------------------------------------------------------------------------------------- |
  | `version`          | `int`        | QRT file version                                                                                |
  | `n_orig`           | `unsigned`   | Number of origin (TX) positions                                                                 |
  | `n_cir`            | `unsigned`   | Number of CIRs per origin                                                                       |
  | `n_dest`           | `unsigned`   | Number of destinations (RX)                                                                     |
  | `n_freq`           | `unsigned`   | Number of frequency bands                                                                       |
  | `freq`             | `arma::fvec` | Frequencies in Hz (converted from GHz for v4 files and older v5 files; v4: of the first origin); `[n_freq]` |
  | `cir_pos`          | `arma::fmat` | CIR positions; `[n_cir, 3]`                                                                     |
  | `cir_orientation`  | `arma::fmat` | CIR orientations (Euler); `[n_cir, 3]`                                                          |
  | `orig_pos_all`     | `arma::fmat` | Origin positions; `[n_orig, 3]`                                                                 |
  | `orig_orientation` | `arma::fmat` | Origin orientations (Euler); `[n_orig, 3]`                                                      |
  | `orig_index`       | `arma::uvec` | Byte offsets from BOF to each origin data block; `[n_orig]`                                     |
  | `path_data_offset` | `arma::uvec` | Absolute offset to path_data_index array per origin; 0 if the TX slot is unwritten; `[n_orig]`  |
  | `orig_freq`        | `arma::fvec` | v4 files only: frequency of each origin in Hz, 0 if the TX slot is unwritten; `[n_orig]`; empty otherwise |
MD!*/

quadriga_lib::qrt_read_cache quadriga_lib::qrt_read_cache_init(const std::string &fn,
                                                               std::ifstream *file)
{
    quadriga_lib::qrt_read_cache cache;

    // Determine which stream to use and whether we own it
    std::ifstream local_stream;
    bool own_stream = (file == nullptr);
    std::ifstream &fileR = own_stream ? local_stream : *file;

    if (own_stream)
    {
        fileR.open(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::runtime_error("Cannot open file '" + fn + "'.");
    }
    const size_t file_size = qrt_file_size(fileR);
    fileR.seekg(0, std::ios::beg);
    if (!own_stream && !fileR.good())
        throw std::runtime_error("Supplied ifstream is not in a good state.");

    cache.version = qrt_read_header(fileR);

    // Global counters
    fileR.read((char *)&cache.n_orig, sizeof(unsigned));
    fileR.read((char *)&cache.n_cir, sizeof(unsigned));
    fileR.read((char *)&cache.n_dest, sizeof(unsigned));

    size_t pos = 11 + 3 * sizeof(unsigned); // read position (ID, version, 3 counters)

    // Version 4 has one frequency, read from the first origin below; 0 Hz if there is no origin
    cache.n_freq = 1;
    cache.freq.zeros(1);
    if (cache.version > 4)
        qrt_read_freq_list(fileR, file_size, pos, cache.n_freq, cache.freq);

    // CIR metadata (read all)
    if (cache.n_cir != 0)
    {
        uint8_t cir_fmt = 0;
        fileR.read((char *)&cir_fmt, sizeof(uint8_t));
        pos += 1;
        size_t cir_bytes = (3 + qrt_orientation_columns(cir_fmt)) * sizeof(float);
        qrt_check_fits(fileR, file_size, pos, cache.n_cir, cir_bytes, "n_cir");
        pos += (size_t)cache.n_cir * cir_bytes;

        cache.cir_pos.set_size(cache.n_cir, 3);
        fileR.read((char *)cache.cir_pos.memptr(), cache.n_cir * 3 * sizeof(float));

        cache.cir_orientation.zeros(cache.n_cir, 3);
        if (cir_fmt > 3)
            fileR.read((char *)cache.cir_orientation.colptr(0), cache.n_cir * sizeof(float));
        if (cir_fmt == 2 || cir_fmt == 3 || cir_fmt == 6 || cir_fmt == 7)
            fileR.read((char *)cache.cir_orientation.colptr(1), cache.n_cir * sizeof(float));
        if (cir_fmt == 1 || cir_fmt == 3 || cir_fmt == 5 || cir_fmt == 7)
            fileR.read((char *)cache.cir_orientation.colptr(2), cache.n_cir * sizeof(float));
        qrt_check_stream(fileR);
    }

    // Skip destination metadata (per destination: CIR offset and name length, at least 5 bytes)
    if (cache.n_dest != 0)
    {
        qrt_check_fits(fileR, file_size, pos, cache.n_dest, sizeof(unsigned) + 1, "n_dest");
        fileR.seekg((std::streamoff)(cache.n_dest * sizeof(unsigned)), std::ios::cur);
        pos += (size_t)cache.n_dest * sizeof(unsigned);
        for (unsigned i = 0; i < cache.n_dest; ++i)
        {
            uint8_t mt_name_length = 0;
            fileR.read((char *)&mt_name_length, sizeof(uint8_t));
            if (mt_name_length != 0)
                fileR.seekg((std::streamoff)mt_name_length, std::ios::cur);
            pos += 1 + (size_t)mt_name_length;
        }
    }

    // Origin metadata (read all)
    if (cache.n_orig != 0)
    {
        uint8_t orig_fmt = 0;
        fileR.read((char *)&orig_fmt, sizeof(uint8_t));
        pos += 1;
        qrt_check_fits(fileR, file_size, pos, cache.n_orig, (3 + qrt_orientation_columns(orig_fmt)) * sizeof(float) + sizeof(size_t), "n_orig");

        cache.orig_pos_all.set_size(cache.n_orig, 3);
        fileR.read((char *)cache.orig_pos_all.memptr(), cache.n_orig * 3 * sizeof(float));

        cache.orig_orientation.zeros(cache.n_orig, 3);
        if (orig_fmt > 3)
            fileR.read((char *)cache.orig_orientation.colptr(0), cache.n_orig * sizeof(float));
        if (orig_fmt == 2 || orig_fmt == 3 || orig_fmt == 6 || orig_fmt == 7)
            fileR.read((char *)cache.orig_orientation.colptr(1), cache.n_orig * sizeof(float));
        if (orig_fmt == 1 || orig_fmt == 3 || orig_fmt == 5 || orig_fmt == 7)
            fileR.read((char *)cache.orig_orientation.colptr(2), cache.n_orig * sizeof(float));

        cache.orig_index.set_size(cache.n_orig);
        fileR.read((char *)cache.orig_index.memptr(), cache.n_orig * sizeof(size_t));
        qrt_check_stream(fileR);
    }

    // path_data_offset: absolute byte offset of path_data_index[] in the data block of each origin, laid out as
    // [tx_name_length(1)] [tx_name(N)] [freq(4, v4 only)] [max_paths(4)] [path_data_index[n_cir]]
    if (cache.n_orig != 0)
    {
        cache.path_data_offset.set_size(cache.n_orig);
        unsigned extra = (cache.version == 4) ? (unsigned)sizeof(float) : 0;
        if (cache.version == 4) // Version 4: one frequency per origin, stored in the origin block before max_paths
            cache.orig_freq.zeros(cache.n_orig);

        for (unsigned i = 0; i < cache.n_orig; ++i)
        {
            if (cache.orig_index(i) == 0) // TX not yet written by append
            {
                cache.path_data_offset(i) = 0;
                continue;
            }
            qrt_check_offset(file_size, cache.orig_index(i), "orig_index");
            fileR.seekg((std::streampos)cache.orig_index(i));
            uint8_t tx_name_length = 0;
            fileR.read((char *)&tx_name_length, sizeof(uint8_t));
            qrt_check_stream(fileR);
            cache.path_data_offset(i) = cache.orig_index(i) + 1  // tx_name_length byte
                                        + (size_t)tx_name_length // tx_name
                                        + (size_t)extra          // freq (v4 only)
                                        + sizeof(unsigned);      // max_paths
            if (cache.version == 4)
            {
                fileR.seekg((std::streamoff)tx_name_length, std::ios::cur);
                fileR.read((char *)cache.orig_freq.memptr() + i * sizeof(float), sizeof(float));
                qrt_check_stream(fileR);
            }

            // The path_data_index of the origin must lie inside the file
            qrt_check_fits(file_size, cache.path_data_offset(i), cache.n_cir, sizeof(size_t), "n_cir");
        }
    }

    // Stored frequencies to Hz; version 4 reports the frequency of the first origin (0 Hz if it is not written yet)
    if (cache.version == 4 && cache.n_orig != 0)
        cache.freq(0) = cache.orig_freq(0);
    cache.freq *= (float)qd_qrt_freq_scale(cache.freq.memptr(), cache.freq.n_elem, cache.version);
    cache.orig_freq *= (float)qd_qrt_freq_scale(cache.orig_freq.memptr(), cache.orig_freq.n_elem, cache.version);

    // Close only if we opened the stream ourselves
    if (own_stream && fileR.is_open())
        fileR.close();

    return cache;
}

/*!MD
# qrt_file_init
Create a new QRT file and write its metadata header

- Writes a v5 (EM) or v6 (scalar) header: frequencies, CIR positions/orientations, RX (MT) metadata, and a reserved BS (TX) region.
- Writes no path data. The BS position/orientation rows and the BS index table are reserved as zeros; each [[qrt_file_append]] call fills the next free slot.
- CIR orientation is stored compressed: only the angle columns that carry a nonzero value after conversion to `float` are written,
  encoded in a per-file format byte. [[qrt_file_append]] reads this byte from the file.
- `n_orig` fixes the number of TX slots; the file can hold at most that many appended TX blocks.
- Positional data is converted to `float` on write regardless of `dtype`.

## Declaration:
```
template <typename dtype>
void quadriga_lib::qrt_file_init(
    const std::string &fn,
    const arma::Col<dtype> &center_freq,
    const arma::Mat<dtype> &cir_pos,
    const arma::Mat<dtype> &cir_orientation,
    const std::vector<std::string> &dest_names = {"RX"},
    const arma::u32_vec &cir_offset = {0},
    unsigned n_orig = 1,
    bool scalar_mode = false);
```

## Inputs:
- **`fn`** — Path to the QRT file to create (truncated if it exists; all inputs are checked before the file is
  opened, so an invalid input throws and leaves an existing file unchanged)
- **`center_freq`** — Center frequencies in Hz, finite and > 0, at least 1 kHz for an EM file; `[n_freq]`, 1 to 127 entries; stored in Hz as `float`
- **`cir_pos`** — CIR positions in Cartesian coordinates; `[n_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles (bank, tilt, head); `[n_cir, 3]` or empty for none
- **`dest_names`** *(optional)* — Receiver (MT) names, at most 255 characters each; `[n_dest]`
- **`cir_offset`** *(optional)* — Number of CIRs before each receiver (offset of its first CIR), the first entry 0; `[n_dest]`, must equal `dest_names` in length
- **`n_orig`** *(optional)* — Number of origin (TX) slots to reserve; at least 1
- **`scalar_mode`** *(optional)* — `true` writes a v6 scalar-layout file, `false` a v5 EM file

## See also:
- [[qrt_file_append]] (write one TX block into a reserved slot)
- [[qrt_file_read]] (read CIR data back)
MD!*/

template <typename dtype>
void quadriga_lib::qrt_file_init(const std::string &fn,
                                 const arma::Col<dtype> &center_freq,
                                 const arma::Mat<dtype> &cir_pos,
                                 const arma::Mat<dtype> &cir_orientation,
                                 const std::vector<std::string> &dest_names,
                                 const arma::u32_vec &cir_offset,
                                 unsigned n_orig,
                                 bool scalar_mode)
{
    if (center_freq.n_elem == 0 || center_freq.n_elem > 127)
        throw std::invalid_argument("Input 'center_freq' must have 1 to 127 elements.");
    if (!center_freq.is_finite() || arma::any(center_freq <= dtype(0)))
        throw std::invalid_argument("Input 'center_freq' must be finite and > 0.");
    if (!scalar_mode && arma::any(center_freq < dtype(1000))) // The readers take a v5 list below 1000 as GHz
        throw std::invalid_argument("Input 'center_freq' must be at least 1 kHz for an EM file.");
    if (n_orig == 0)
        throw std::invalid_argument("Input 'n_orig' must be at least 1.");

    // All inputs are checked here, before the file is opened, so that a rejected call leaves an existing file unchanged
    if (cir_pos.n_cols != 3)
        throw std::invalid_argument("Input 'cir_pos' must have size [n_cir, 3].");
    if ((unsigned long long)cir_pos.n_rows > (unsigned long long)std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Input 'cir_pos' has too many rows.");
    unsigned n_cir = (unsigned)cir_pos.n_rows;
    if (cir_orientation.n_elem != 0 && (cir_orientation.n_rows != n_cir || cir_orientation.n_cols != 3))
        throw std::invalid_argument("Input 'cir_orientation' must be empty or have size [n_cir, 3].");
    if (dest_names.size() != cir_offset.n_elem)
        throw std::invalid_argument("Inputs 'dest_names' and 'cir_offset' must have the same length.");
    if ((unsigned long long)dest_names.size() > (unsigned long long)std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Input 'dest_names' has too many entries.");
    if (cir_offset.n_elem != 0 && cir_offset[0] != 0)
        throw std::invalid_argument("Input 'cir_offset' must start with 0.");
    for (const auto &name : dest_names)
        if (name.size() > 255)
            throw std::invalid_argument("Entries of 'dest_names' cannot exceed 255 characters.");

    unsigned n_dest = (unsigned)dest_names.size();
    unsigned n_freq = (unsigned)center_freq.n_elem;

    // Write a dtype column as float, one element at a time.
    auto wr_col = [](std::ofstream &f, const dtype *src, unsigned n)
    {
        for (unsigned i = 0; i < n; ++i)
        {
            float v = (float)src[i];
            f.write((char *)&v, sizeof(float));
        }
    };

    // Compressed orientation format: set a bit only where a column carries a nonzero angle.
    // Column order is [bank, tilt, head]; bit2 = bank, bit1 = tilt, bit0 = head.
    // The test uses the stored float values, so an angle that rounds to 0.0f does not set a bit.
    auto derive_fmt = [](const arma::Mat<dtype> &o, unsigned n) -> uint8_t
    {
        if (o.n_elem == 0 || o.n_rows != n || o.n_cols < 3)
            return 0;
        auto nz = [&](unsigned c)
        { const dtype *p = o.colptr(c); for (unsigned i = 0; i < n; ++i) if ((float)p[i] != 0.0f) return true; return false; };
        uint8_t f = 0;
        if (nz(0))
            f |= 0x4;
        if (nz(1))
            f |= 0x2;
        if (nz(2))
            f |= 0x1;
        return f;
    };

    std::ofstream fileW(fn, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!fileW.is_open())
        throw std::runtime_error("Cannot create file '" + fn + "'.");

    std::string bin_id = scalar_mode ? "#QRT-BINv06" : "#QRT-BINv05";
    fileW.write(bin_id.c_str(), (std::streamsize)bin_id.size());

    fileW.write((char *)&n_orig, sizeof(unsigned));
    fileW.write((char *)&n_cir, sizeof(unsigned));
    fileW.write((char *)&n_dest, sizeof(unsigned));
    fileW.write((char *)&n_freq, sizeof(unsigned));

    wr_col(fileW, center_freq.memptr(), n_freq);

    if (n_cir != 0)
    {
        uint8_t cir_fmt = derive_fmt(cir_orientation, n_cir);
        fileW.write((char *)&cir_fmt, sizeof(uint8_t));
        wr_col(fileW, cir_pos.colptr(0), n_cir);
        wr_col(fileW, cir_pos.colptr(1), n_cir);
        wr_col(fileW, cir_pos.colptr(2), n_cir);
        if (cir_fmt & 0x4)
            wr_col(fileW, cir_orientation.colptr(0), n_cir);
        if (cir_fmt & 0x2)
            wr_col(fileW, cir_orientation.colptr(1), n_cir);
        if (cir_fmt & 0x1)
            wr_col(fileW, cir_orientation.colptr(2), n_cir);
    }

    fileW.write((char *)cir_offset.memptr(), n_dest * sizeof(unsigned));
    for (unsigned i = 0; i < n_dest; ++i)
    {
        uint8_t len = (uint8_t)dest_names[i].size();
        fileW.write((char *)&len, sizeof(uint8_t));
        fileW.write(dest_names[i].data(), (std::streamsize)len);
    }

    uint8_t orig_fmt = 7;
    fileW.write((char *)&orig_fmt, sizeof(uint8_t));
    std::vector<float> zeros_pos(6 * n_orig, 0.0f);
    fileW.write((char *)zeros_pos.data(), (std::streamsize)(zeros_pos.size() * sizeof(float)));

    std::vector<size_t> zero_index(n_orig, 0);
    fileW.write((char *)zero_index.data(), (std::streamsize)(zero_index.size() * sizeof(size_t)));

    fileW.close();
    if (!fileW)
        throw std::runtime_error("Cannot write file '" + fn + "'.");
}

template void quadriga_lib::qrt_file_init<float>(const std::string &fn, const arma::Col<float> &center_freq, const arma::Mat<float> &cir_pos,
                                                 const arma::Mat<float> &cir_orientation, const std::vector<std::string> &dest_names,
                                                 const arma::u32_vec &cir_offset, unsigned n_orig, bool scalar_mode);

template void quadriga_lib::qrt_file_init<double>(const std::string &fn, const arma::Col<double> &center_freq, const arma::Mat<double> &cir_pos,
                                                  const arma::Mat<double> &cir_orientation, const std::vector<std::string> &dest_names,
                                                  const arma::u32_vec &cir_offset, unsigned n_orig, bool scalar_mode);

/*!MD
# qrt_file_append
Append one transmitter's path data to an existing QRT file

- Writes the path data for a single TX into the next free slot reserved by [[qrt_file_init]], and records the TX name, position, and orientation.
- Paths are grouped by their CIR index (`path::i_cir`) and stored per CIR as: interaction counts, polarization coefficients, interaction coordinates, and interaction type codes.
- Validates every path before writing: `i_cir` must be within the file's CIR count, and each path's frequency count and layout mode (EM/scalar) must match the file exactly.
- Throws when all TX slots are already filled — the file holds at most the `n_orig` slots reserved at init.
- Only v5/6 files can be appended to. Positional data is converted to `float` on write.
- Returns the total number of paths written.

## Declaration:
```
template <typename dtype>
size_t quadriga_lib::qrt_file_append(
    const std::string &fn,
    const std::vector<quadriga_lib::path> &path_data,
    const arma::Mat<dtype> &orig_pos,
    const arma::Mat<dtype> &orig_orientation = arma::Mat<dtype>(),
    const std::string &orig_name = "TX");
```

## Inputs:
- **`fn`** — Path to an existing QRT file created by [[qrt_file_init]]
- **`path_data`** — Paths to write; each carries its CIR index in `i_cir` and must match the file's frequency count and layout mode
- **`orig_pos`** — Transmitter position; `[3, 1]` or `[1, 3]`
- **`orig_orientation`** *(optional)* — Transmitter orientation (bank, tilt, heading) in rad; `[3, 1]` or `[1, 3]`; empty = zeros
- **`orig_name`** *(optional)* — Transmitter name; at most 255 characters

## Returns:
- Number of paths written across all CIRs.

## See also:
- [[qrt_file_init]] (create the file and reserve TX slots)
- [[qrt_file_read]] (read the appended data back)
MD!*/

template <typename dtype>
size_t quadriga_lib::qrt_file_append(const std::string &fn,
                                     const std::vector<quadriga_lib::path> &path_data,
                                     const arma::Mat<dtype> &orig_pos,
                                     const arma::Mat<dtype> &orig_orientation,
                                     const std::string &orig_name)
{
    const dtype zeros[3] = {dtype(0), dtype(0), dtype(0)};
    const dtype *p_pos = qd_xyz(orig_pos, "orig_pos");
    const dtype *p_ori = !orig_orientation.empty() ? qd_xyz(orig_orientation, "orig_orientation") : zeros;
    if (orig_name.size() > 255)
        throw std::invalid_argument("Input 'orig_name' cannot exceed 255 characters.");

    std::fstream f(fn, std::ios::in | std::ios::out | std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file '" + fn + "'.");

    // Recover metadata via the existing header parser (open the file read-only for it).
    quadriga_lib::qrt_read_cache cache;
    {
        std::ifstream fr(fn, std::ios::in | std::ios::binary);
        if (!fr.is_open())
            throw std::runtime_error("Cannot open file '" + fn + "'.");
        cache = quadriga_lib::qrt_read_cache_init(fn, &fr);
    }

    if (cache.version != 5 && cache.version != 6)
        throw std::runtime_error("Only QRT versions 5 and 6 can be appended to.");

    unsigned n_cir = cache.n_cir;
    unsigned n_freq = cache.n_freq;
    unsigned n_orig = cache.n_orig;
    bool scalar = (cache.version == 6);
    size_t pol_stride = scalar ? 2 : 8;

    // Origin (BS) block: with orig_fmt = 7 it is orig_fmt(1) + 6 * n_orig floats, followed by the u64 index table;
    // orig_pos_base is the start of the position columns, orig_index_base the start of the index table. The header
    // is replayed up to orig_fmt to find them.
    size_t off = 11 + 4 * sizeof(unsigned); // ID + n_orig/n_cir/n_dest/n_freq
    off += (size_t)n_freq * sizeof(float);  // frequencies[]
    if (n_cir != 0)
    {
        // cir_fmt byte + 3 pos columns + orientation columns per cir_fmt; the byte is read from the file
        uint8_t cir_fmt = 0;
        f.seekg((std::streamoff)off, std::ios::beg);
        f.read((char *)&cir_fmt, sizeof(uint8_t));
        if (!f)
            throw std::runtime_error("QRT file truncated or corrupt.");
        size_t cols = 3 + qrt_orientation_columns(cir_fmt);
        off += 1 + cols * n_cir * sizeof(float);
    }
    // MT index + names (variable-length names): walk them from the file.
    {
        f.seekg((std::streamoff)off, std::ios::beg);
        f.seekg((std::streamoff)(cache.n_dest * sizeof(unsigned)), std::ios::cur);
        for (unsigned i = 0; i < cache.n_dest; ++i)
        {
            uint8_t len = 0;
            f.read((char *)&len, sizeof(uint8_t));
            f.seekg((std::streamoff)len, std::ios::cur);
        }
        if (!f)
            throw std::runtime_error("QRT file truncated or corrupt.");
        off = (size_t)f.tellg();
    }
    size_t orig_pos_base = off + 1;                                       // skip orig_fmt byte
    size_t orig_index_base = orig_pos_base + 6 * n_orig * sizeof(float);  // after 6 columns

    // Validate paths and bin them by CIR; throws before any write
    std::vector<unsigned> no_hit(n_cir, 0);
    for (const auto &p : path_data)
    {
        if ((unsigned)p.i_cir >= n_cir)
            throw std::invalid_argument("Entries of 'path_data' must have a CIR index 'i_cir' below the number of CIRs in the file.");
        if ((unsigned)p.n_freq() != n_freq)
            throw std::invalid_argument("Entries of 'path_data' must have as many frequencies as the file.");
        if (p.is_scalar() != scalar)
            throw std::invalid_argument("Entries of 'path_data' must match the scalar or polarimetric mode of the file.");
        ++no_hit[p.i_cir];
    }
    unsigned max_paths = 0;
    for (unsigned c = 0; c < n_cir; ++c)
        if (no_hit[c] > max_paths)
            max_paths = no_hit[c];

    std::vector<std::vector<const quadriga_lib::path *>> by_cir(n_cir);
    for (unsigned c = 0; c < n_cir; ++c)
        by_cir[c].reserve(no_hit[c]);
    for (const auto &p : path_data)
        by_cir[p.i_cir].push_back(&p);

    // First free BS slot (0 = free); throw if full.
    unsigned slot = n_orig;
    f.seekg((std::streamoff)orig_index_base, std::ios::beg);
    for (unsigned i = 0; i < n_orig; ++i)
    {
        size_t v = 0;
        f.read((char *)&v, sizeof(size_t));
        if (!f)
            throw std::runtime_error("QRT file truncated or corrupt.");
        if (v == 0)
        {
            slot = i;
            break;
        }
    }
    if (slot == n_orig)
        throw std::runtime_error("All origin slots of the file are written.");

    // Patch this TX's position/orientation row (columns x,y,z,bank,tilt,head; orig_fmt = 7).
    auto patch_col = [&](unsigned col, float v)
    {
        size_t o = orig_pos_base + size_t(col * n_orig + slot) * sizeof(float);
        f.seekp((std::streamoff)o, std::ios::beg);
        f.write((char *)&v, sizeof(float));
    };
    patch_col(0, (float)p_pos[0]);
    patch_col(1, (float)p_pos[1]);
    patch_col(2, (float)p_pos[2]);
    patch_col(3, (float)p_ori[0]);
    patch_col(4, (float)p_ori[1]);
    patch_col(5, (float)p_ori[2]);

    // Append the bs_data_t block at EOF.
    f.seekp(0, std::ios::end);
    size_t block_off = (size_t)f.tellp();

    uint8_t name_len = (uint8_t)orig_name.size();
    f.write((char *)&name_len, sizeof(uint8_t));
    f.write(orig_name.data(), (std::streamsize)name_len);
    f.write((char *)&max_paths, sizeof(unsigned));

    std::streamoff index_pos = (std::streamoff)f.tellp();
    std::vector<size_t> path_index(n_cir, 0);
    f.write((char *)path_index.data(), (std::streamsize)(path_index.size() * sizeof(size_t)));

    size_t total_paths = 0;
    for (unsigned c = 0; c < n_cir; ++c)
    {
        path_index[c] = (size_t)f.tellp();
        const auto &paths = by_cir[c];
        unsigned nP = (unsigned)paths.size();
        f.write((char *)&nP, sizeof(unsigned));

        std::vector<uint8_t> nmi(nP);
        size_t sum_int = 0;
        for (unsigned i = 0; i < nP; ++i)
        {
            size_t ns = paths[i]->n_seg();
            uint8_t v = ns > 255 ? (uint8_t)255 : (uint8_t)ns;
            nmi[i] = v;
            sum_int += v;
        }
        if (nP != 0)
            f.write((char *)nmi.data(), (std::streamsize)nP);

        if (nP != 0)
        {
            std::vector<float> pol(pol_stride * nP * n_freq);
            for (unsigned i_freq = 0; i_freq < n_freq; ++i_freq)
                for (unsigned iP = 0; iP < nP; ++iP)
                {
                    const float *cf = paths[iP]->polmat(i_freq);
                    float *dst = pol.data() + size_t(i_freq * nP + iP) * pol_stride;
                    for (size_t k = 0; k < pol_stride; ++k)
                        dst[k] = cf[k];
                }
            f.write((char *)pol.data(), (std::streamsize)(pol.size() * sizeof(float)));
        }

        for (unsigned iP = 0; iP < nP; ++iP)
            if (nmi[iP] != 0)
                f.write((char *)paths[iP]->coord(0), (std::streamsize)((size_t)nmi[iP] * 3 * sizeof(float)));

        if (sum_int != 0)
        {
            std::vector<uint8_t> types;
            types.reserve(sum_int);
            for (unsigned iP = 0; iP < nP; ++iP)
            {
                if (nmi[iP] == 0)
                    continue;
                std::vector<uint8_t> seq = paths[iP]->interaction_type_codes(); // size == n_seg()
                types.insert(types.end(), seq.begin(), seq.begin() + nmi[iP]);
            }
            f.write((char *)types.data(), (std::streamsize)types.size());
        }

        total_paths += nP;
    }

    size_t eof = (size_t)f.tellp();
    f.seekp(index_pos, std::ios::beg);
    f.write((char *)path_index.data(), (std::streamsize)(path_index.size() * sizeof(size_t)));
    f.seekp((std::streamoff)(orig_index_base + (size_t)slot * sizeof(size_t)), std::ios::beg);
    f.write((char *)&block_off, sizeof(size_t));
    f.seekp((std::streamoff)eof, std::ios::beg);

    f.close();
    if (!f)
        throw std::runtime_error("Cannot write file '" + fn + "'.");
    return total_paths;
}

template size_t quadriga_lib::qrt_file_append<float>(const std::string &fn, const std::vector<quadriga_lib::path> &path_data,
                                                     const arma::Mat<float> &orig_pos, const arma::Mat<float> &orig_orientation,
                                                     const std::string &orig_name);

template size_t quadriga_lib::qrt_file_append<double>(const std::string &fn, const std::vector<quadriga_lib::path> &path_data,
                                                      const arma::Mat<double> &orig_pos, const arma::Mat<double> &orig_orientation,
                                                      const std::string &orig_name);

/*!MD
# qrt_file_parse
Read metadata from a QRT file

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count, CIR offsets, names, positions, orientations, and file version.
- All output arguments are optional; pass `nullptr` to skip any.
- If `file` is `nullptr`, the file is opened internally and closed on return; if provided, the stream is left open.
- When `n_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and `cir_offset` reflect this.
- Throws `std::runtime_error` if the file cannot be opened, is not a QRT file of a supported version, is truncated,
  or if a count or offset in it does not fit the file size.

## Declaration:
```
void quadriga_lib::qrt_file_parse(
    const std::string &fn,
    arma::uword *n_cir = nullptr,
    arma::uword *n_orig = nullptr,
    arma::uword *n_dest = nullptr,
    arma::uword *n_freq = nullptr,
    arma::uvec *cir_offset = nullptr,
    std::vector<std::string> *orig_names = nullptr,
    std::vector<std::string> *dest_names = nullptr,
    int *version = nullptr,
    arma::fvec *center_freq = nullptr,
    arma::fmat *cir_pos = nullptr,
    arma::fmat *cir_orientation = nullptr,
    arma::fmat *orig_pos = nullptr,
    arma::fmat *orig_orientation = nullptr,
    std::ifstream *file = nullptr);
```

## Inputs:
- **`fn`** — Path to the QRT file
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally

## Outputs:
- **`n_cir`** *(optional)* — Number of channel snapshots per origin point
- **`n_orig`** *(optional)* — Number of origin points (TX)
- **`n_dest`** *(optional)* — Number of destination points (RX)
- **`n_freq`** *(optional)* — Number of frequency bands
- **`cir_offset`** *(optional)* — Number of CIRs before each destination (offset of its first CIR), the first entry 0; `[n_dest]`
- **`orig_names`** *(optional)* — Names of origin points; `[n_orig]`
- **`dest_names`** *(optional)* — Names of destination points; `[n_dest]`
- **`version`** *(optional)* — QRT file version number
- **`center_freq`** *(optional)* — Center frequencies in Hz, converted from GHz for files that store GHz (v4, and v5 when all stored values are below 1000); `[n_freq]`
- **`cir_pos`** *(optional)* — CIR positions in Cartesian coordinates; `[n_cir, 3]`
- **`cir_orientation`** *(optional)* — CIR orientations as Euler angles; `[n_cir, 3]`
- **`orig_pos`** *(optional)* — Origin (TX) positions in Cartesian coordinates; `[n_orig, 3]`
- **`orig_orientation`** *(optional)* — Origin (TX) orientations as Euler angles; `[n_orig, 3]`
MD!*/

void quadriga_lib::qrt_file_parse(const std::string &fn,
                                  arma::uword *n_cir,
                                  arma::uword *n_orig,
                                  arma::uword *n_dest,
                                  arma::uword *n_freq,
                                  arma::uvec *cir_offset,
                                  std::vector<std::string> *orig_names,
                                  std::vector<std::string> *dest_names,
                                  int *version,
                                  arma::fvec *center_freq,
                                  arma::fmat *cir_pos,
                                  arma::fmat *cir_orientation,
                                  arma::fmat *orig_pos,
                                  arma::fmat *orig_orientation,
                                  std::ifstream *file)
{
    qd_check_alias_list({n_cir, n_orig, n_dest, n_freq}, {"n_cir", "n_orig", "n_dest", "n_freq"});
    qd_check_alias_list({orig_names, dest_names}, {"orig_names", "dest_names"});
    qd_check_alias_list<arma::fmat>({center_freq, cir_pos, cir_orientation, orig_pos, orig_orientation},
                                    {"center_freq", "cir_pos", "cir_orientation", "orig_pos", "orig_orientation"});

    // Determine which stream to use and whether we own it
    std::ifstream local_stream;
    bool own_stream = (file == nullptr);

    std::ifstream &fileR = own_stream ? local_stream : *file;

    if (own_stream)
    {
        fileR.open(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::runtime_error("Cannot open file '" + fn + "'.");
    }

    // Rewind the stream to the beginning
    const size_t file_size = qrt_file_size(fileR);
    fileR.seekg(0, std::ios::beg);
    if (!own_stream && !fileR.good())
        throw std::runtime_error("Supplied ifstream is not in a good state.");

    int ver = qrt_read_header(fileR);

    // Global counters
    unsigned l_no_orig = 0, l_no_cir = 0, l_no_dest = 0, l_no_freq = 1;

    fileR.read((char *)&l_no_orig, sizeof(unsigned));
    fileR.read((char *)&l_no_cir, sizeof(unsigned));
    fileR.read((char *)&l_no_dest, sizeof(unsigned));

    size_t pos = 11 + 3 * sizeof(unsigned); // read position (ID, version, 3 counters)

    arma::fvec l_freq(1, arma::fill::zeros); // Version 4: read from the first origin below; 0 Hz if there is no origin
    if (ver > 4)
        qrt_read_freq_list(fileR, file_size, pos, l_no_freq, l_freq);

    // CIR metadata
    arma::fmat l_cir_pos;
    arma::fmat l_cir_orientation;
    uint8_t cir_fmt = 0;

    if (l_no_cir != 0)
    {
        fileR.read((char *)&cir_fmt, sizeof(uint8_t));
        pos += 1;
        size_t cir_bytes = (3 + qrt_orientation_columns(cir_fmt)) * sizeof(float);
        qrt_check_fits(fileR, file_size, pos, l_no_cir, cir_bytes, "n_cir");
        pos += (size_t)l_no_cir * cir_bytes;

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

    // Destination (RX) metadata
    arma::u32_vec l_cir_index;
    std::vector<std::string> l_dest_names;

    if (l_no_dest != 0)
    {
        // Per destination: CIR offset and name length, at least 5 bytes
        qrt_check_fits(fileR, file_size, pos, l_no_dest, sizeof(unsigned) + 1, "n_dest");
        l_cir_index.set_size(l_no_dest);
        fileR.read((char *)l_cir_index.memptr(), l_no_dest * sizeof(unsigned));
        pos += (size_t)l_no_dest * sizeof(unsigned);

        l_dest_names.resize((size_t)l_no_dest);
        for (unsigned i = 0; i < l_no_dest; ++i)
        {
            uint8_t mt_name_length = 0;
            fileR.read((char *)&mt_name_length, sizeof(uint8_t));
            l_dest_names[i].resize((size_t)mt_name_length);
            fileR.read(&l_dest_names[i][0], (size_t)mt_name_length);
            pos += 1 + (size_t)mt_name_length;
        }
        qrt_check_stream(fileR);
    }
    else
    {
        l_cir_index.zeros(1);
        l_dest_names.resize(1);
        l_dest_names[0] = "RX";
    }

    // Origin (TX) metadata
    arma::fmat l_orig_pos_all;
    arma::fmat l_orig_orientation;
    arma::uvec l_orig_index;

    if (l_no_orig != 0)
    {
        uint8_t orig_fmt = 0;
        fileR.read((char *)&orig_fmt, sizeof(uint8_t));
        pos += 1;
        qrt_check_fits(fileR, file_size, pos, l_no_orig, (3 + qrt_orientation_columns(orig_fmt)) * sizeof(float) + sizeof(size_t), "n_orig");

        l_orig_pos_all.set_size(l_no_orig, 3);
        fileR.read((char *)l_orig_pos_all.memptr(), l_no_orig * 3 * sizeof(float));

        l_orig_orientation.zeros(l_no_orig, 3);
        if (orig_fmt > 3) // Bank angle
            fileR.read((char *)l_orig_orientation.colptr(0), l_no_orig * sizeof(float));
        if (orig_fmt == 2 || orig_fmt == 3 || orig_fmt == 6 || orig_fmt == 7) // Tilt angle
            fileR.read((char *)l_orig_orientation.colptr(1), l_no_orig * sizeof(float));
        if (orig_fmt == 1 || orig_fmt == 3 || orig_fmt == 5 || orig_fmt == 7) // Heading angle
            fileR.read((char *)l_orig_orientation.colptr(2), l_no_orig * sizeof(float));

        l_orig_index.set_size(l_no_orig);
        fileR.read((char *)l_orig_index.memptr(), l_no_orig * sizeof(size_t));
    }
    qrt_check_stream(fileR);

    if (l_cir_index.n_elem == 0 || l_cir_index[0] != 0)
        throw std::runtime_error("Invalid CIR index in QRT file. Potential file corruption.");

    // Read origin names — requires seeking to each origin's data block
    std::vector<std::string> l_orig_names;
    if (orig_names)
    {
        l_orig_names.reserve(l_no_orig);
        for (unsigned i = 0; i < l_no_orig; ++i)
        {
            if (l_orig_index(i) == 0) // TX slot reserved but not yet written by append
            {
                l_orig_names.push_back(std::string());
                continue;
            }
            qrt_check_offset(file_size, l_orig_index(i), "orig_index");
            fileR.seekg((std::streampos)l_orig_index(i));

            uint8_t tx_name_length = 0;
            fileR.read((char *)&tx_name_length, sizeof(uint8_t));

            std::string name((size_t)tx_name_length, '\0');
            fileR.read(&name[0], (size_t)tx_name_length);
            qrt_check_stream(fileR);

            l_orig_names.push_back(std::move(name));
        }
    }

    if (center_freq && ver == 4 && l_no_orig != 0)
    {
        // In version 4 the frequency is stored per-origin; read from the first origin
        qrt_check_offset(file_size, l_orig_index(0), "orig_index");
        fileR.seekg((std::streampos)l_orig_index(0));

        uint8_t tx_name_length = 0;
        fileR.read((char *)&tx_name_length, sizeof(uint8_t));
        fileR.seekg((std::streamoff)tx_name_length, std::ios::cur); // skip name

        l_freq.set_size(1);
        fileR.read((char *)l_freq.memptr(), sizeof(float));
        qrt_check_stream(fileR);
    }

    // Populate output parameters, after all reads and checks
    if (n_cir)
        *n_cir = (arma::uword)l_no_cir;
    if (n_orig)
        *n_orig = (arma::uword)l_no_orig;
    if (n_dest)
        *n_dest = (arma::uword)(l_no_dest == 0 ? 1 : l_no_dest);
    if (n_freq)
        *n_freq = (arma::uword)l_no_freq;
    if (version)
        *version = ver;

    if (cir_offset)
    {
        cir_offset->set_size(l_cir_index.n_elem);
        auto po = cir_offset->memptr();
        for (auto &val : l_cir_index)
            *po++ = (arma::uword)val;
    }

    if (dest_names)
        *dest_names = std::move(l_dest_names);
    if (orig_names)
        *orig_names = std::move(l_orig_names);

    if (center_freq)
    {
        l_freq *= (float)qd_qrt_freq_scale(l_freq.memptr(), l_freq.n_elem, ver);
        *center_freq = std::move(l_freq);
    }

    if (cir_pos)
        *cir_pos = std::move(l_cir_pos);
    if (cir_orientation)
        *cir_orientation = std::move(l_cir_orientation);
    if (orig_pos)
        *orig_pos = std::move(l_orig_pos_all);
    if (orig_orientation)
        *orig_orientation = std::move(l_orig_orientation);

    // Close only if we opened the stream ourselves
    if (own_stream && fileR.is_open())
        fileR.close();
}

/*!MD
# qrt_file_read
Read ray-tracing CIR data from a QRT file

- Reads the channel impulse response data of a list of CIRs (snapshots) of one origin point.
- All output arguments are optional; pass `nullptr` to skip any. Per-CIR outputs are vectors with one entry per
  requested CIR; positions and orientations have one column per requested CIR.
- If `uplink = false`, origin is TX and destination is RX; if `true`, roles are swapped and `polmat` is the transpose
  of the downlink `polmat` (reciprocal channel, no conjugation) with the same `path_gain`; `n_interact`, `coord` and
  `interact_type` keep the stored order (origin to destination) in both directions, and the type codes describe
  that traversal.
- Without a `cache`, each call reads the whole file header first. To read a file in parts (e.g. in chunks of CIRs,
  to bound the memory), pass a pre-opened `std::ifstream` and a [[qrt_read_cache_init]]-populated cache; the
  per-CIR I/O is then 2 seeks and 4 reads. `fn` is ignored when both `file` and `cache` are provided.
- Reading a TX slot that was reserved by [[qrt_file_init]] but not yet written by [[qrt_file_append]] returns
  empty path outputs with zeroed positions, rather than throwing.
- Path data is read only when a path output (`fbs_pos` to `interact_type`) is requested; positions and orientations
  come from the header.
- Throws `std::invalid_argument` for an index out of range, before any CIR is read or any output is changed. Throws `std::runtime_error` if
  the file cannot be opened, is not a QRT file of a supported version, is truncated, or if a count or offset in it
  does not fit the file size (path data: only when it is read); outputs may then be partly written. A message about
  one requested CIR starts with `"Snapshot <k>: "`, with k its position in `i_cir`.

## Declaration:
```
void quadriga_lib::qrt_file_read(
    const std::string &fn,
    const arma::uvec &i_cir = {},
    arma::uword i_orig = 0,
    bool uplink = false,
    arma::Col<dtype> *center_freq = nullptr,
    arma::Mat<dtype> *tx_pos = nullptr,
    arma::Mat<dtype> *tx_orientation = nullptr,
    arma::Mat<dtype> *rx_pos = nullptr,
    arma::Mat<dtype> *rx_orientation = nullptr,
    std::vector<arma::Mat<dtype>> *fbs_pos = nullptr,
    std::vector<arma::Mat<dtype>> *lbs_pos = nullptr,
    std::vector<arma::Mat<dtype>> *path_gain = nullptr,
    std::vector<arma::Col<dtype>> *path_length = nullptr,
    std::vector<arma::Cube<dtype>> *polmat = nullptr,
    std::vector<arma::Col<dtype>> *aod = nullptr,
    std::vector<arma::Col<dtype>> *eod = nullptr,
    std::vector<arma::Col<dtype>> *aoa = nullptr,
    std::vector<arma::Col<dtype>> *eoa = nullptr,
    std::vector<std::vector<arma::Mat<dtype>>> *path_coord = nullptr,
    int normalize_polmat = 1,
    std::vector<arma::u32_vec> *n_interact = nullptr,
    std::vector<arma::fmat> *coord = nullptr,
    std::vector<std::vector<uint8_t>> *interact_type = nullptr,
    std::ifstream *file = nullptr,
    const qrt_read_cache *cache = nullptr);
```

## Inputs:
- **`fn`** — Path to the QRT file; ignored when both `file` and `cache` are provided
- **`i_cir`** *(optional)* — CIR (snapshot) indices, 0-based; `[n_out]`; empty = all CIRs of the file (`n_out = 0` for a file
  without CIRs)
- **`i_orig`** *(optional)* — Origin index, 0-based
- **`uplink`** *(optional)* — If `false`, origin=TX, destination=RX; if `true`, roles are swapped (origin=RX, destination=TX)
- **`normalize_polmat`** *(optional)* — Controls `polmat` and `path_gain` scaling where PL is the propagation-only path loss
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm<br><br>

    | `normalize_polmat` | `polmat`                                   | `path_gain`                      |
    | ------------- | ----------------------------------------------- | -------------------------------- |
    | 0             | As stored in QRT file                           | -PL                              |
    | 1             | Max column power = 1; uplink: max row power = 1 | -PL minus material losses        |<br><br>

  Other values throw `std::invalid_argument`.
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; left open on return
- **`cache`** *(optional)* — Pre-populated cache from [[qrt_read_cache_init]]<br><br>

## Outputs:
- **`center_freq`** *(optional)* — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** *(optional)* — Transmitter positions in Cartesian coordinates; `[3, n_out]`
- **`tx_orientation`** *(optional)* — Transmitter orientations (bank, tilt, heading); `[3, n_out]`
- **`rx_pos`** *(optional)* — Receiver positions in Cartesian coordinates; `[3, n_out]`
- **`rx_orientation`** *(optional)* — Receiver orientations (bank, tilt, heading); `[3, n_out]`
- **`fbs_pos`** *(optional)* — First-bounce scatterer positions; `[n_out]` of `[3, n_path]`
- **`lbs_pos`** *(optional)* — Last-bounce scatterer positions; `[n_out]` of `[3, n_path]`
- **`path_gain`** *(optional)* — Path gain on linear scale; `[n_out]` of `[n_path, n_freq]`
- **`path_length`** *(optional)* — Absolute path length TX to RX phase center; `[n_out]` of `[n_path]`
- **`polmat`** *(optional)* — Polarization transfer matrix; `[n_out]` of `[8, n_path, n_freq]`, or `[2, n_path, n_freq]` for v6 files
- **`aod`** *(optional)* — Departure azimuth angles; `[n_out]` of `[n_path]`
- **`eod`** *(optional)* — Departure elevation angles; `[n_out]` of `[n_path]`
- **`aoa`** *(optional)* — Arrival azimuth angles; `[n_out]` of `[n_path]`
- **`eoa`** *(optional)* — Arrival elevation angles; `[n_out]` of `[n_path]`
- **`path_coord`** *(optional)* — Interaction coordinates per path; `[n_out]` of `[n_path]` of `[3, n_interact + 2]`
- **`n_interact`** *(optional)* — Number of mesh interactions per path; 0 indicates LOS; `[n_out]` of `[n_path]`
- **`coord`** *(optional)* — Interaction coordinates; `[n_out]` of `[3, sum(n_interact)]`
- **`interact_type`** *(optional)* — Interaction type codes, concatenated per path and segmented by `n_interact`; `[n_out]` of `[sum(n_interact)]`.
  Empty for v4 legacy files. GBSM paths of [[qrt_paths_from_scatterers]] carry 129 (GR) and 193 (NLOS); LOS and GR
  paths are not marked as GBSM.

## Example:
```
// Read all CIRs of origin 0 in chunks of 256, so that only one chunk is in memory at a time
std::ifstream stream("scene.qrt", std::ios::in | std::ios::binary);
auto cache = quadriga_lib::qrt_read_cache_init("scene.qrt", &stream);
std::vector<arma::mat> path_gain;
std::vector<arma::vec> path_length;
for (arma::uword i0 = 0; i0 < cache.n_cir; i0 += 256)
{
    arma::uvec i_cir = arma::regspace<arma::uvec>(i0, std::min(i0 + 256, (arma::uword)cache.n_cir) - 1);
    quadriga_lib::qrt_file_read<double>("", i_cir, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, &path_gain, &path_length, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1,
        nullptr, nullptr, nullptr, &stream, &cache);
    // ... use path_gain[k], path_length[k] of CIR i_cir[k]
}
```

## See also:
- [[qrt_read_cache_init]] (populate cache for reading a file in parts)
- [[qrt_file_parse]] (extract file metadata without reading CIR data)
MD!*/

// Reads one CIR, with the file metadata from the cache; the stream is open and the indices are checked
// - Positions and orientations are written to 3 consecutive elements each (nullptr = not requested)
template <typename dtype>
static void qrt_read_cir(std::ifstream &fileR, size_t file_size, const quadriga_lib::qrt_read_cache &c,
                         arma::uword i_cir, arma::uword i_orig, bool uplink, int normalize_polmat,
                         const arma::vec &gain_at_1m, const arma::vec &alpha_dB_per_m,
                         dtype *tx_pos, dtype *tx_orientation, dtype *rx_pos, dtype *rx_orientation,
                         arma::Mat<dtype> *fbs_pos, arma::Mat<dtype> *lbs_pos, arma::Mat<dtype> *path_gain,
                         arma::Col<dtype> *path_length, arma::Cube<dtype> *polmat,
                         arma::Col<dtype> *aod, arma::Col<dtype> *eod, arma::Col<dtype> *aoa, arma::Col<dtype> *eoa,
                         std::vector<arma::Mat<dtype>> *path_coord, arma::u32_vec *n_interact, arma::fmat *coord,
                         std::vector<uint8_t> *interact_type)
{
    const int ver = c.version;
    const unsigned l_no_freq = c.n_freq;
    const arma::uword n_freq = (arma::uword)l_no_freq;
    const bool v6 = (ver == 6);
    const size_t l_path_data_offset = (size_t)c.path_data_offset(i_orig);

    // Origin (O) and destination (D) position and orientation; uplink swaps TX and RX
    const dtype Ox = (dtype)c.orig_pos_all(i_orig, 0), Oy = (dtype)c.orig_pos_all(i_orig, 1), Oz = (dtype)c.orig_pos_all(i_orig, 2);
    const dtype Dx = (dtype)c.cir_pos(i_cir, 0), Dy = (dtype)c.cir_pos(i_cir, 1), Dz = (dtype)c.cir_pos(i_cir, 2);
    const dtype O[3] = {Ox, Oy, Oz}, D[3] = {Dx, Dy, Dz};
    const dtype Oo[3] = {(dtype)c.orig_orientation(i_orig, 0), (dtype)c.orig_orientation(i_orig, 1), (dtype)c.orig_orientation(i_orig, 2)};
    const dtype Do[3] = {(dtype)c.cir_orientation(i_cir, 0), (dtype)c.cir_orientation(i_cir, 1), (dtype)c.cir_orientation(i_cir, 2)};

    // TX slot reserved but not yet appended: zero positions, no paths
    const bool unwritten = (l_path_data_offset == 0);
    const dtype Z[3] = {(dtype)0, (dtype)0, (dtype)0};
    auto put = [unwritten, &Z](dtype *out, const dtype *val)
    {
        if (out)
            std::memcpy(out, unwritten ? Z : val, 3 * sizeof(dtype));
    };
    put(tx_pos, uplink ? D : O);
    put(rx_pos, uplink ? O : D);
    put(tx_orientation, uplink ? Do : Oo);
    put(rx_orientation, uplink ? Oo : Do);

    // No path output requested: positions and orientations come from the cache, the path data is not read
    if (!(fbs_pos || lbs_pos || path_gain || path_length || polmat || aod || eod || aoa || eoa || path_coord ||
          n_interact || coord || interact_type))
        return;

    if (unwritten)
    {
        if (n_interact)
            n_interact->reset();
        if (coord)
            coord->reset();
        if (interact_type)
            interact_type->clear();
        if (polmat)
            polmat->reset();
        if (path_gain)
            path_gain->reset();
        if (path_length)
            path_length->reset();
        if (aod)
            aod->reset();
        if (eod)
            eod->reset();
        if (aoa)
            aoa->reset();
        if (eoa)
            eoa->reset();
        if (fbs_pos)
            fbs_pos->reset();
        if (lbs_pos)
            lbs_pos->reset();
        if (path_coord)
            path_coord->clear();
        return;
    }

    // Read CIR path data (2 seeks + 4 reads)
    // Seek directly to path_data_index[i_cir] within the origin block
    fileR.seekg((std::streampos)(l_path_data_offset + i_cir * sizeof(size_t)));

    size_t data_offset = 0;
    fileR.read((char *)&data_offset, sizeof(size_t));
    qrt_check_stream(fileR);
    qrt_check_offset(file_size, data_offset, "path_data_index");

    // Seek to the actual CIR data block
    fileR.seekg((std::streampos)data_offset);

    // Number of paths
    unsigned n_path = 0;
    fileR.read((char *)&n_path, sizeof(unsigned));
    qrt_check_stream(fileR);

    // Each path needs its interaction count and its polarization coefficients
    size_t xprmat_size = v6 ? 2 : 8;
    size_t data_pos = data_offset + sizeof(unsigned);
    size_t path_bytes = 1 + xprmat_size * (size_t)l_no_freq * sizeof(float);
    qrt_check_fits(file_size, data_pos, n_path, path_bytes, "n_path");
    data_pos += (size_t)n_path * path_bytes;

    // Number of mesh interactions per path
    arma::u32_vec no_intR;
    arma::uword sum_no_int = 0;
    no_intR.set_size(n_path);
    {
        unsigned *p_no_int = no_intR.memptr();
        for (unsigned iP = 0; iP < n_path; ++iP)
        {
            uint8_t no_mesh_interact_byte = 0;
            fileR.read((char *)&no_mesh_interact_byte, sizeof(uint8_t));
            p_no_int[iP] = (unsigned)no_mesh_interact_byte;
            sum_no_int += p_no_int[iP];
        }
    }

    // Each interaction needs its coordinates and, in v5/v6, its type code
    qrt_check_fits(file_size, data_pos, sum_no_int, 3 * sizeof(float) + (ver != 4 ? 1 : 0), "sum(n_interact)");

    // Polarization transfer matrix
    arma::fcube xprmatR(xprmat_size, n_path, l_no_freq);
    fileR.read((char *)xprmatR.memptr(), xprmat_size * n_path * l_no_freq * sizeof(float));

    // Interaction coordinates
    arma::fmat coordR(3, sum_no_int);
    fileR.read((char *)coordR.memptr(), 3 * sum_no_int * sizeof(float));

    // Interaction type codes (v5/v6 only; absent in v4 legacy files)
    std::vector<uint8_t> interact_typeR;
    if (ver != 4)
    {
        interact_typeR.resize(sum_no_int);
        uint8_t *p_it = interact_typeR.data();
        for (arma::uword k = 0; k < sum_no_int; ++k)
        {
            uint8_t b = 0;
            fileR.read((char *)&b, sizeof(uint8_t));
            p_it[k] = b;
        }
    }

    qrt_check_stream(fileR);

    // Output: raw interaction data
    if (n_interact)
        *n_interact = no_intR;

    if (coord)
        *coord = coordR;

    if (interact_type)
        *interact_type = interact_typeR;

    // Calculate path gain and polarization matrix polmat
    // - xprmatR includes all interaction losses, but not the path loss
    // - here we calculate the normalized polarization matrix polmat and the PG without path loss
    if (polmat || path_gain)
    {
        dtype *dst = nullptr;
        if (polmat)
        {
            if (v6)
                polmat->set_size(2, n_path, n_freq);
            else
                polmat->set_size(8, n_path, n_freq);
            dst = polmat->memptr();
        }

        dtype *pg = nullptr;
        if (path_gain)
        {
            path_gain->set_size(n_path, n_freq);
            pg = path_gain->memptr();
        }

        const float *src = xprmatR.memptr();

        for (arma::uword i_freq = 0; i_freq < n_freq; ++i_freq)
            for (arma::uword i_path = 0; i_path < n_path; ++i_path)
            {
                const dtype r11 = (dtype)src[0];
                const dtype i11 = (dtype)src[1];
                const dtype r21 = v6 ? dtype(0) : (dtype)src[2];
                const dtype i21 = v6 ? dtype(0) : (dtype)src[3];
                const dtype r12 = v6 ? dtype(0) : (dtype)src[4];
                const dtype i12 = v6 ? dtype(0) : (dtype)src[5];
                const dtype r22 = v6 ? dtype(0) : (dtype)src[6];
                const dtype i22 = v6 ? dtype(0) : (dtype)src[7];

                // column powers (V and H) - path gain = max column power
                dtype gain = 1.0;
                if (normalize_polmat == 1)
                {
                    const dtype p1 = r11 * r11 + i11 * i11 + r21 * r21 + i21 * i21;
                    const dtype p2 = r12 * r12 + i12 * i12 + r22 * r22 + i22 * i22;
                    gain = (p1 > p2) ? p1 : p2;
                }

                if (pg)
                    *pg++ = gain;

                // normalization factor: max column power -> 1
                dtype scale = (dtype)0;
                if (gain > (dtype)0)
                    scale = dtype(1.0 / std::sqrt((double)gain));

                if (dst)
                {
                    if (!uplink) // copy, normalize
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
                    else // uplink: transpose, normalize
                    {
                        // Reciprocal medium: H_UL = H_DL^T, no conjugation (TR 38.901 clause 7.5)
                        // The scale stays that of the downlink matrix, so path_gain is the same in both directions
                        dst[0] = r11 * scale;
                        dst[1] = i11 * scale;
                        if (!v6)
                        {
                            dst[2] = r12 * scale;
                            dst[3] = i12 * scale;
                            dst[4] = r21 * scale;
                            dst[5] = i21 * scale;
                            dst[6] = r22 * scale;
                            dst[7] = i22 * scale;
                        }
                    }
                    dst += v6 ? 2 : 8;
                }
                src += v6 ? 2 : 8;
            }
    }

    bool want_angles = aod || eod || aoa || eoa;
    bool want_length = path_gain || path_length;

    // Extract path metadata
    // - here we add the path loss from path length to the PG
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

        const arma::Col<dtype> o_pos = {Ox, Oy, Oz}, d_pos = {Dx, Dy, Dz};
        if (want_angles && want_length)
            quadriga_lib::coord2path<dtype>(o_pos, d_pos, no_intR, coordD,
                                            &path_length_local, fbs_pos, lbs_pos, &path_angles, path_coord, uplink);
        else if (want_angles && !want_length)
            quadriga_lib::coord2path<dtype>(o_pos, d_pos, no_intR, coordD,
                                            nullptr, fbs_pos, lbs_pos, &path_angles, path_coord, uplink);
        else if (!want_angles && want_length)
            quadriga_lib::coord2path<dtype>(o_pos, d_pos, no_intR, coordD,
                                            &path_length_local, fbs_pos, lbs_pos, nullptr, path_coord, uplink);
        else // want_none
            quadriga_lib::coord2path<dtype>(o_pos, d_pos, no_intR, coordD,
                                            nullptr, fbs_pos, lbs_pos, nullptr, path_coord, uplink);

        // Adjust path gain to include the path loss
        if (want_length)
        {
            if (path_length)
                path_length->set_size(n_path);

            dtype *src = path_length_local.memptr();
            dtype *pg = path_gain ? path_gain->memptr() : nullptr;
            dtype *pl = path_length ? path_length->memptr() : nullptr;
            const double *p_gain_at_1m = gain_at_1m.memptr();
            const double *p_alpha = alpha_dB_per_m.memptr();

            for (arma::uword i_path = 0; i_path < n_path; ++i_path)
            {
                dtype len = src[i_path];
                if (pl)
                    pl[i_path] = len;
                if (pg)
                {
                    for (arma::uword i_freq = 0; i_freq < n_freq; ++i_freq)
                    {
                        double gainPL = p_gain_at_1m[i_freq] - 20.0 * std::log10((double)len) - p_alpha[i_freq] * (double)len;
                        pg[i_freq * n_path + i_path] *= (dtype)std::pow(10.0, 0.1 * gainPL);
                    }
                }
            }
        }

        // Copy the angles (no memcpy for 0 paths: the empty matrices have no data pointer)
        if (aod)
        {
            aod->set_size(n_path);
            if (n_path)
                std::memcpy(aod->memptr(), path_angles.colptr(0), n_path * sizeof(dtype));
        }

        if (eod)
        {
            eod->set_size(n_path);
            if (n_path)
                std::memcpy(eod->memptr(), path_angles.colptr(1), n_path * sizeof(dtype));
        }

        if (aoa)
        {
            aoa->set_size(n_path);
            if (n_path)
                std::memcpy(aoa->memptr(), path_angles.colptr(2), n_path * sizeof(dtype));
        }

        if (eoa)
        {
            eoa->set_size(n_path);
            if (n_path)
                std::memcpy(eoa->memptr(), path_angles.colptr(3), n_path * sizeof(dtype));
        }
    }
}

template <typename dtype>
void quadriga_lib::qrt_file_read(const std::string &fn, const arma::uvec &i_cir, arma::uword i_orig, bool uplink,
                                 arma::Col<dtype> *center_freq, arma::Mat<dtype> *tx_pos, arma::Mat<dtype> *tx_orientation,
                                 arma::Mat<dtype> *rx_pos, arma::Mat<dtype> *rx_orientation,
                                 std::vector<arma::Mat<dtype>> *fbs_pos, std::vector<arma::Mat<dtype>> *lbs_pos,
                                 std::vector<arma::Mat<dtype>> *path_gain, std::vector<arma::Col<dtype>> *path_length,
                                 std::vector<arma::Cube<dtype>> *polmat,
                                 std::vector<arma::Col<dtype>> *aod, std::vector<arma::Col<dtype>> *eod,
                                 std::vector<arma::Col<dtype>> *aoa, std::vector<arma::Col<dtype>> *eoa,
                                 std::vector<std::vector<arma::Mat<dtype>>> *path_coord, int normalize_polmat,
                                 std::vector<arma::u32_vec> *n_interact, std::vector<arma::fmat> *coord,
                                 std::vector<std::vector<uint8_t>> *interact_type,
                                 std::ifstream *file, const qrt_read_cache *cache)
{
    if (normalize_polmat != 0 && normalize_polmat != 1)
        throw std::invalid_argument("Input 'normalize_polmat' must be 0 or 1.");

    // Outputs must be distinct and must not be part of the cache (possible for dtype = float)
    if (cache)
        qd_check_alias_list<arma::Mat<dtype>>({center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation},
                                              {"center_freq", "tx_pos", "tx_orientation", "rx_pos", "rx_orientation"},
                                              &cache->freq, &cache->cir_pos, &cache->cir_orientation, &cache->orig_pos_all,
                                              &cache->orig_orientation, &cache->orig_freq);
    else
        qd_check_alias_list<arma::Mat<dtype>>({center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation},
                                              {"center_freq", "tx_pos", "tx_orientation", "rx_pos", "rx_orientation"});
    qd_check_alias_list<std::vector<arma::Mat<dtype>>>({fbs_pos, lbs_pos, path_gain}, {"fbs_pos", "lbs_pos", "path_gain"});
    qd_check_alias_list<std::vector<arma::Col<dtype>>>({path_length, aod, eod, aoa, eoa}, {"path_length", "aod", "eod", "aoa", "eoa"});
    qd_check_alias_outputs(coord, "coord", fbs_pos, "fbs_pos"); // Same type for dtype = float
    qd_check_alias_outputs(coord, "coord", lbs_pos, "lbs_pos");
    qd_check_alias_outputs(coord, "coord", path_gain, "path_gain");

    // Stream setup
    std::ifstream local_stream;
    bool own_stream = (file == nullptr);
    std::ifstream &fileR = own_stream ? local_stream : *file;

    if (own_stream)
    {
        fileR.open(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::runtime_error("Cannot open file '" + fn + "'.");
    }

    // Without a cache, the whole header is read once for all requested CIRs
    qrt_read_cache local_cache;
    if (cache == nullptr)
    {
        local_cache = qrt_read_cache_init(fn, &fileR);
        cache = &local_cache;
    }
    const qrt_read_cache &c = *cache;

    for (arma::uword k = 0; k < i_cir.n_elem; ++k)
        if (i_cir[k] >= (arma::uword)c.n_cir)
            throw std::invalid_argument("Snapshot " + std::to_string(k) + ": Input 'i_cir' is out of range; the file has " +
                                        std::to_string(c.n_cir) + " CIRs.");
    const arma::uword n_out = i_cir.empty() ? (arma::uword)c.n_cir : i_cir.n_elem;

    // A file without origins (possible for v4) reads empty when no CIR is requested
    if (c.n_orig != 0 && i_orig >= (arma::uword)c.n_orig)
        throw std::invalid_argument("Input 'i_orig' is out of range; the file has " + std::to_string(c.n_orig) + " origins.");
    if (n_out != 0 && c.n_orig == 0) // qrt_file_init writes at least one origin
        throw std::runtime_error("File does not contain any origins.");

    const size_t file_size = qrt_file_size(fileR);
    const arma::uword n_freq = (arma::uword)c.n_freq;
    const bool v6 = (c.version == 6);

    // Version 4 stores one frequency per origin
    const float *p_freq = (c.version == 4 && i_orig < (arma::uword)c.orig_freq.n_elem) ? &c.orig_freq.at(i_orig) : c.freq.memptr();

    // Gain at 1 m (path-loss reference); EM: FSPL reference, scalar: 1/r spreading, frequency-independent
    // Air absorption in dB/m (ISO 9613-1, T = 20 C, RH = 50 %, p = 1 atm), zero for v4/v5 (EM)
    arma::vec gain_at_1m(n_freq), alpha_dB_per_m(n_freq, arma::fill::zeros);
    for (arma::uword i_freq = 0; i_freq < n_freq; ++i_freq)
    {
        gain_at_1m[i_freq] = qd_gain_at_1m((double)p_freq[i_freq], v6);
        if (v6)
            alpha_dB_per_m[i_freq] = qd_iso9613_alpha((double)p_freq[i_freq]);
    }

    if (center_freq)
    {
        center_freq->set_size(n_freq);
        for (arma::uword i_freq = 0; i_freq < n_freq; ++i_freq)
            center_freq->at(i_freq) = (dtype)p_freq[i_freq];
    }

    // Output sizes
    for (arma::Mat<dtype> *p : {tx_pos, tx_orientation, rx_pos, rx_orientation})
        if (p)
            p->set_size(3, n_out);
    auto resize = [n_out](auto *v)
    {
        if (v)
            v->resize(n_out);
    };
    resize(fbs_pos), resize(lbs_pos), resize(path_gain), resize(path_length), resize(polmat);
    resize(aod), resize(eod), resize(aoa), resize(eoa), resize(path_coord);
    resize(n_interact), resize(coord), resize(interact_type);

    auto col = [](arma::Mat<dtype> *m, arma::uword k)
    { return m ? m->colptr(k) : nullptr; };
    auto at = [](auto *v, arma::uword k)
    { return v ? &(*v)[k] : nullptr; };

    // Stops at the first failing CIR; its error gets the prefix "Snapshot <k>: " (k = position in the list)
    for (arma::uword k = 0; k < n_out; ++k)
        try
        {
            qrt_read_cir<dtype>(fileR, file_size, c, i_cir.empty() ? k : i_cir[k], i_orig, uplink, normalize_polmat,
                                gain_at_1m, alpha_dB_per_m, col(tx_pos, k), col(tx_orientation, k), col(rx_pos, k),
                                col(rx_orientation, k), at(fbs_pos, k), at(lbs_pos, k), at(path_gain, k), at(path_length, k),
                                at(polmat, k), at(aod, k), at(eod, k), at(aoa, k), at(eoa, k), at(path_coord, k),
                                at(n_interact, k), at(coord, k), at(interact_type, k));
        }
        catch (const std::invalid_argument &e)
        {
            throw std::invalid_argument("Snapshot " + std::to_string(k) + ": " + e.what());
        }
        catch (const std::runtime_error &e)
        {
            throw std::runtime_error("Snapshot " + std::to_string(k) + ": " + e.what());
        }

    // Close only if we opened the stream ourselves
    if (own_stream && fileR.is_open())
        fileR.close();
}

template void quadriga_lib::qrt_file_read(const std::string &fn, const arma::uvec &i_cir, arma::uword i_orig, bool uplink,
                                          arma::Col<float> *center_freq, arma::Mat<float> *tx_pos, arma::Mat<float> *tx_orientation,
                                          arma::Mat<float> *rx_pos, arma::Mat<float> *rx_orientation,
                                          std::vector<arma::Mat<float>> *fbs_pos, std::vector<arma::Mat<float>> *lbs_pos,
                                          std::vector<arma::Mat<float>> *path_gain, std::vector<arma::Col<float>> *path_length,
                                          std::vector<arma::Cube<float>> *polmat,
                                          std::vector<arma::Col<float>> *aod, std::vector<arma::Col<float>> *eod,
                                          std::vector<arma::Col<float>> *aoa, std::vector<arma::Col<float>> *eoa,
                                          std::vector<std::vector<arma::Mat<float>>> *path_coord, int normalize_polmat,
                                          std::vector<arma::u32_vec> *n_interact, std::vector<arma::fmat> *coord,
                                          std::vector<std::vector<uint8_t>> *interact_type,
                                          std::ifstream *file, const qrt_read_cache *cache);

template void quadriga_lib::qrt_file_read(const std::string &fn, const arma::uvec &i_cir, arma::uword i_orig, bool uplink,
                                          arma::Col<double> *center_freq, arma::Mat<double> *tx_pos, arma::Mat<double> *tx_orientation,
                                          arma::Mat<double> *rx_pos, arma::Mat<double> *rx_orientation,
                                          std::vector<arma::Mat<double>> *fbs_pos, std::vector<arma::Mat<double>> *lbs_pos,
                                          std::vector<arma::Mat<double>> *path_gain, std::vector<arma::Col<double>> *path_length,
                                          std::vector<arma::Cube<double>> *polmat,
                                          std::vector<arma::Col<double>> *aod, std::vector<arma::Col<double>> *eod,
                                          std::vector<arma::Col<double>> *aoa, std::vector<arma::Col<double>> *eoa,
                                          std::vector<std::vector<arma::Mat<double>>> *path_coord, int normalize_polmat,
                                          std::vector<arma::u32_vec> *n_interact, std::vector<arma::fmat> *coord,
                                          std::vector<std::vector<uint8_t>> *interact_type,
                                          std::ifstream *file, const qrt_read_cache *cache);

/*!MD
# qrt_file_read_raw
Read raw ray-tracing path data from a QRT file into path objects

- Reassembles the stored path data for a single origin (TX) into a vector of [[path]] objects — the inverse of [[qrt_file_append]].
- Unlike [[qrt_file_read]], which returns processed CIR data (channel matrices, angles, path loss), this returns the raw per-path storage:
  coordinates, polarization coefficients, and interaction type codes.
- Each returned path has `i_cir` set to its CIR index. Counters not stored in the file
  (`n_ref`, `n_tra`, `n_sub`, `n_sct`) and `length` are left at their defaults.
- Only v05/v06 files can be read as raw paths; v4 legacy files are rejected.
- An origin slot that was reserved by [[qrt_file_init]] but never written by [[qrt_file_append]] returns an empty vector.
- Throws `std::runtime_error` if the file cannot be opened, is not a QRT file of a supported version, is truncated,
  or if a count or offset in it does not fit the file size.
- For tight loops over many origins, pass a pre-opened `std::ifstream` and a [[qrt_read_cache_init]]-populated cache to avoid
  re-parsing the header on each call.

## Declaration:
```
std::vector<quadriga_lib::path> quadriga_lib::qrt_file_read_raw(
    const std::string &fn,
    arma::uword i_orig = 0,
    std::ifstream *file = nullptr,
    const qrt_read_cache *cache = nullptr);
```

## Inputs:
- **`fn`** — Path to the QRT file; ignored when both `file` and `cache` are supplied
- **`i_orig`** *(optional)* — Origin index to read (for downlink, origin = TX); must be less than `n_orig`
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally
- **`cache`** *(optional)* — Pre-parsed metadata from [[qrt_read_cache_init]]; pass `nullptr` to parse the header on this call

## Returns:
- Vector of [[path]] objects for the requested origin, one per stored path across all CIRs; empty if the origin slot is unwritten.

## See also:
- [[qrt_file_append]] (write path data — the inverse operation)
- [[qrt_read_cache_init]] (populate cache for fast repeated reads)
- [[qrt_file_read]] (read processed CIR data instead of raw paths)
MD!*/

std::vector<quadriga_lib::path> quadriga_lib::qrt_file_read_raw(const std::string &fn,
                                                                arma::uword i_orig,
                                                                std::ifstream *file,
                                                                const qrt_read_cache *cache)
{
    // Stream: use the supplied one, or open our own.
    std::ifstream local_stream;
    bool own_stream = (file == nullptr);
    std::ifstream &fileR = own_stream ? local_stream : *file;

    if (own_stream)
    {
        fileR.open(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::runtime_error("Cannot open file '" + fn + "'.");
    }
    else if (!fileR.good())
        throw std::runtime_error("Supplied ifstream is not in a good state.");

    // Metadata: use the supplied cache, or build a local one.
    quadriga_lib::qrt_read_cache local_cache;
    const quadriga_lib::qrt_read_cache *c = cache;
    if (c == nullptr)
    {
        local_cache = quadriga_lib::qrt_read_cache_init(fn, &fileR);
        c = &local_cache;
    }

    if (c->version != 5 && c->version != 6)
        throw std::runtime_error("Only QRT versions 5 and 6 can be read as raw paths.");
    if (i_orig >= (arma::uword)c->n_orig)
        throw std::invalid_argument("Input 'i_orig' is out of range; the file has " + std::to_string(c->n_orig) + " origins.");

    unsigned n_cir = c->n_cir;
    unsigned n_freq = c->n_freq;
    bool scalar = (c->version == 6);
    size_t stride = scalar ? 2 : 8;

    std::vector<quadriga_lib::path> out;

    size_t pdo = c->path_data_offset((arma::uword)i_orig);
    if (pdo == 0) // Origin slot reserved but never written by append
    {
        if (own_stream && fileR.is_open())
            fileR.close();
        return out;
    }

    // Per-CIR path_data_index for this origin (n_cir u64 offsets).
    const size_t file_size = qrt_file_size(fileR);
    qrt_check_fits(file_size, pdo, n_cir, sizeof(size_t), "n_cir");
    std::vector<size_t> cir_offset(n_cir);
    fileR.seekg((std::streampos)pdo, std::ios::beg);
    fileR.read((char *)cir_offset.data(), (std::streamsize)(n_cir * sizeof(size_t)));
    qrt_check_stream(fileR);

    for (unsigned ci = 0; ci < n_cir; ++ci)
    {
        qrt_check_offset(file_size, cir_offset[ci], "path_data_index");
        fileR.seekg((std::streampos)cir_offset[ci], std::ios::beg);

        unsigned nP = 0;
        fileR.read((char *)&nP, sizeof(unsigned));
        qrt_check_stream(fileR);
        if (nP == 0)
            continue;

        // Each path needs its interaction count and its polarization coefficients
        size_t data_pos = cir_offset[ci] + sizeof(unsigned);
        size_t path_bytes = 1 + stride * (size_t)n_freq * sizeof(float);
        qrt_check_fits(file_size, data_pos, nP, path_bytes, "n_path");
        data_pos += (size_t)nP * path_bytes;

        // no_mesh_interact[nP] = n_seg() per path
        std::vector<uint8_t> nmi(nP);
        fileR.read((char *)nmi.data(), (std::streamsize)nP);
        qrt_check_stream(fileR);
        size_t sum_int = 0;
        for (unsigned i = 0; i < nP; ++i)
            sum_int += nmi[i];

        // Each interaction needs its coordinates and its type code
        qrt_check_fits(file_size, data_pos, sum_int, 3 * sizeof(float) + 1, "sum(n_interact)");

        // polarization[stride * nP * n_freq], freq-major then path
        std::vector<float> pol(stride * nP * n_freq);
        fileR.read((char *)pol.data(), (std::streamsize)(pol.size() * sizeof(float)));

        // mesh_interact[3 * sum] coordinates, concatenated per path
        std::vector<float> coords(3 * sum_int);
        if (sum_int != 0)
            fileR.read((char *)coords.data(), (std::streamsize)(coords.size() * sizeof(float)));

        // interaction_type[sum] codes, concatenated per path
        std::vector<uint8_t> types(sum_int);
        if (sum_int != 0)
            fileR.read((char *)types.data(), (std::streamsize)sum_int);

        if (!fileR)
            throw std::runtime_error("QRT file truncated or corrupt: unexpected end of file in path block.");

        // Reassemble one path object per stored path.
        // Grow the output geometrically: a reserve of exactly the next size per CIR would move all paths read so far
        size_t coord_cursor = 0, type_cursor = 0;
        if (out.size() + nP > out.capacity())
            out.reserve(std::max(out.size() + nP, 2 * out.capacity()));
        for (unsigned iP = 0; iP < nP; ++iP)
        {
            unsigned ns = nmi[iP];

            quadriga_lib::path p(ns, n_freq, scalar); // allocates buffer, seeds identity
            p.i_cir = ci;

            // Coordinates -> coord buffer (3 * ns floats)
            if (ns != 0)
            {
                std::memcpy(p.coord(0), coords.data() + coord_cursor * 3, (size_t)ns * 3 * sizeof(float));
                coord_cursor += ns;
            }

            // Polarization -> polmat slots, undoing the freq-major interleave
            for (unsigned i_freq = 0; i_freq < n_freq; ++i_freq)
            {
                const float *src = pol.data() + ((size_t)i_freq * nP + iP) * stride;
                std::memcpy(p.polmat(i_freq), src, stride * sizeof(float));
            }

            // Interaction type codes -> inline array + history buffer
            if (ns != 0)
            {
                p.set_interaction_type_codes(std::vector<uint8_t>(types.begin() + type_cursor, types.begin() + type_cursor + ns));
                type_cursor += ns;
            }
            out.push_back(std::move(p));
        }
    }

    if (own_stream && fileR.is_open())
        fileR.close();

    return out;
}

/*!MD
# qrt_paths_from_scatterers
Path objects of one CIR from spherical path data, for qrt_file_append

- Converts the paths of one Tx-Rx pair (e.g. of [[ssf_paths]] with `drift` mobility) into [[path]] objects for
  [[qrt_file_append]]; one call per CIR, the objects of all CIRs of one origin appended together
- Interactions: LOS none; GR one reflection at `lbs_pos` (code 129, `n_ref` = 1); NLOS two point scatterers at
  `fbs_pos`, `lbs_pos` (code 193, `n_sct` = 2); `length` = polyline from the Tx to the last interaction. Counters
  and `length` exist on the returned objects only ([[qrt_file_read_raw]] returns 0): files store the codes
- The path gain and the inverse free-space loss of the polyline are folded into the stored Jones matrices, so that
  [[qrt_file_read]] (`uplink = false`, `normalize_polmat` 0 or 1) returns sqrt(`path_gain`)·`polmat` per complex entry
  within float32 accuracy, the polyline as `path_length` and `path_type` as `n_interact`
- `tx_pos` must be the `orig_pos` of [[qrt_file_append]] and `rx_pos` the CIR position of [[qrt_file_init]]:
  `qrt_file_read` recomputes the lengths from the stored positions; not checked
- Float32 storage: use local coordinates, shifted in x and y only (ground at z = 0); path length error 2.7 µm,
  0.31 mm, 1.0 mm for coordinates up to 150 m, 5 km, 20 km (phase 0.003, 0.39, 1.29 rad at 60 GHz)
- Memory: 80 to 105 bytes per path object at one frequency, e.g. 3 to 4 GB for 1000 UTs, 100 snapshots, 381 rays
- Throws for a path of length 0 in float32 (e.g. LOS with `tx_pos` = `rx_pos`) and for values beyond float32

## Declaration:
```
std::vector<path> quadriga_lib::qrt_paths_from_scatterers(
    const arma::vec &center_freq,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::mat &fbs_pos,
    const arma::mat &lbs_pos,
    const arma::mat &path_gain,
    const arma::cube &polmat,
    const arma::u32_vec &path_type,
    arma::uword i_cir = 0);
```

## Inputs:
- **`center_freq`** — Frequencies in [Hz], 1 to 127 entries; `[n_freq]`
- **`tx_pos`** — Tx (origin) position in [m]; `[3, 1]` or `[1, 3]`
- **`rx_pos`** — Rx position of the CIR in [m]; `[3, 1]` or `[1, 3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in [m] (GR: `lbs_pos`); `[3, n_path]`
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`polmat`** — Polarization transfer matrices, row order of [[get_channels_spherical]]; `[8, n_path, n_freq]`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS, at most one LOS and one GR; `[n_path]`
- **`i_cir`** *(optional)* — CIR index, 0-based, stored in `path::i_cir`, < 2^32

## Returns:
- Path objects in input order, EM layout with `n_freq` frequencies; `[n_path]`

## See also:
- [[qrt_file_append]] (writes the paths)
- [[gbsm_rays_from_paths]] (ray data from the paths read back)
MD!*/

std::vector<quadriga_lib::path> quadriga_lib::qrt_paths_from_scatterers(const arma::vec &center_freq,
                                                                        const arma::mat &tx_pos,
                                                                        const arma::mat &rx_pos,
                                                                        const arma::mat &fbs_pos,
                                                                        const arma::mat &lbs_pos,
                                                                        const arma::mat &path_gain,
                                                                        const arma::cube &polmat,
                                                                        const arma::u32_vec &path_type,
                                                                        arma::uword i_cir)
{
    const arma::uword n_freq = center_freq.n_elem, n_path = fbs_pos.n_cols;
    if (n_freq == 0 || n_freq > 127)
        throw std::invalid_argument("Input 'center_freq' must have 1 to 127 elements.");
    if (!center_freq.is_finite() || arma::any(center_freq <= 0.0))
        throw std::invalid_argument("Input 'center_freq' must be finite and > 0.");
    qd_xyz(tx_pos, "tx_pos"), qd_xyz(rx_pos, "rx_pos");
    if (fbs_pos.n_rows != 3 || lbs_pos.n_rows != 3 || lbs_pos.n_cols != n_path)
        throw std::invalid_argument("Inputs 'fbs_pos' and 'lbs_pos' must have size [3, n_path].");
    if (!fbs_pos.is_finite() || !lbs_pos.is_finite())
        throw std::invalid_argument("Inputs 'fbs_pos' and 'lbs_pos' must be finite.");
    if (path_gain.n_rows != n_path || path_gain.n_cols != n_freq)
        throw std::invalid_argument("Input 'path_gain' must have size [n_path, n_freq].");
    if (!path_gain.is_finite() || arma::any(arma::vectorise(path_gain) < 0.0))
        throw std::invalid_argument("Input 'path_gain' must be finite and >= 0.");
    if (polmat.n_rows != 8 || polmat.n_cols != n_path || polmat.n_slices != n_freq)
        throw std::invalid_argument("Input 'polmat' must have size [8, n_path, n_freq].");
    if (!polmat.is_finite())
        throw std::invalid_argument("Input 'polmat' must be finite.");
    if (path_type.n_elem != n_path)
        throw std::invalid_argument("Input 'path_type' must have n_path elements.");
    if (arma::any(path_type > 2))
        throw std::invalid_argument("Values of 'path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS).");
    if (arma::accu(path_type == 0) > 1 || arma::accu(path_type == 1) > 1) // gbsm_rays_from_paths rejects them
        throw std::invalid_argument("Input 'path_type' can hold at most one LOS and one GR path.");
    if (i_cir > 4294967295ULL)
        throw std::invalid_argument("Input 'i_cir' must be < 2^32.");

    // Polyline lengths in double: LOS the direct line, GR via its point in lbs_pos, NLOS T -> FBS -> LBS -> R
    auto dist = [](const double *a, const double *b) { return qd_calc_length(a[0], a[1], a[2], b[0], b[1], b[2]); };
    const double *T = tx_pos.memptr(), *R = rx_pos.memptr();
    arma::vec L(n_path);
    for (arma::uword p = 0; p < n_path; ++p)
    {
        const double *F = fbs_pos.colptr(p), *B = lbs_pos.colptr(p);
        L(p) = (path_type(p) == 0)   ? dist(T, R)
               : (path_type(p) == 1) ? dist(T, B) + dist(B, R)
                                     : dist(T, F) + dist(F, B) + dist(B, R);
    }

    // Amplitude of the inverse free-space loss at 1 m that qrt_file_read applies, per frequency
    arma::vec a_1m(n_freq);
    for (arma::uword f = 0; f < n_freq; ++f)
        a_1m(f) = std::pow(10.0, -0.05 * qd_gain_at_1m(center_freq(f), false));

    const float Tf[3] = {(float)T[0], (float)T[1], (float)T[2]}, Rf[3] = {(float)R[0], (float)R[1], (float)R[2]};
    std::vector<quadriga_lib::path> out(n_path);
    for (arma::uword p = 0; p < n_path; ++p)
    {
        quadriga_lib::path &q = out[p];
        const arma::uword type = path_type(p);
        q.init(type == 0 ? 0 : type == 1 ? 1 : 2, n_freq);
        q.i_cir = (unsigned)i_cir;
        if (type == 1) // GR: one front reflection at its point
        {
            for (int i = 0; i < 3; ++i)
                q.coord(0)[i] = (float)lbs_pos(i, p);
            q.set_interaction_type_codes({129});
            q.n_ref = 1;
        }
        else if (type == 2) // NLOS: two point scatterers
        {
            for (int i = 0; i < 3; ++i)
                q.coord(0)[i] = (float)fbs_pos(i, p), q.coord(1)[i] = (float)lbs_pos(i, p);
            q.set_interaction_type_codes({193, 193});
            q.n_sct = 2;
        }
        // The stored float32 values must give qrt_file_read a length > 0 and finite Jones matrices
        const float len = q.calc_length(Rf, Tf);
        if (len == 0.0f) // the free-space loss of a zero length is infinite
            throw std::invalid_argument("Inputs 'tx_pos' and 'rx_pos' must differ for a path of zero length.");
        if (!std::isfinite(len))
            throw std::invalid_argument("Inputs 'tx_pos', 'rx_pos', 'fbs_pos' and 'lbs_pos' must be within the float32 "
                                        "range of QRT files.");

        // Stored Jones matrix: polmat·sqrt(path_gain)·L·a_1m, so that qrt_file_read returns sqrt(path_gain)·polmat
        for (arma::uword f = 0; f < n_freq; ++f)
        {
            const double s = std::sqrt(path_gain(p, f)) * L(p) * a_1m(f);
            const double *m = polmat.slice_colptr(f, p);
            float *J = q.polmat(f);
            for (int e = 0; e < 8; ++e)
            {
                J[e] = (float)(m[e] * s);
                if (!std::isfinite(J[e]))
                    throw std::invalid_argument("Inputs 'path_gain' and 'polmat' give stored Jones matrices beyond the "
                                                "float32 range of QRT files.");
            }
        }
    }
    return out;
}
