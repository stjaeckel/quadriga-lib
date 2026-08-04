// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"

static_assert(sizeof(arma::uword) == sizeof(unsigned long long), "arma::uword and unsigned long long have different sizes");
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t and unsigned long long have different sizes");

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# qrt_read_cache_init
Initialize a QRT read cache for fast repeated access

- Reads all fixed metadata from a QRT file into a `quadriga_lib::qrt_read_cache` struct.
- Pre-computes byte offsets so subsequent [[qrt_file_read]] calls need only 2 seeks and 4 reads instead of re-parsing the header.
- Populate once, then pass the cache and a shared `std::ifstream` to [[qrt_file_read]] for tight-loop performance.
- If `file` is `nullptr`, the file is opened internally and closed on return; if provided, the stream is left open.
- For a TX slot reserved by [[qrt_file_init]] but not yet written, `orig_index` and `path_data_offset`
  are `0`; [[qrt_file_read]] treats that as an empty (unwritten) origin.

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
  | `no_orig`          | `unsigned`   | Number of origin (TX) positions                                                                 |
  | `no_cir`           | `unsigned`   | Number of CIRs per origin                                                                       |
  | `no_dest`          | `unsigned`   | Number of destinations (RX)                                                                     |
  | `no_freq`          | `unsigned`   | Number of frequency bands                                                                       |
  | `freq`             | `arma::fvec` | Frequency in GHz; `[no_freq]`                                                                   |
  | `cir_pos`          | `arma::fmat` | CIR positions; `[no_cir, 3]`                                                                    |
  | `cir_orientation`  | `arma::fmat` | CIR orientations (Euler); `[no_cir, 3]`                                                         |
  | `orig_pos_all`     | `arma::fmat` | Origin positions; `[no_orig, 3]`                                                                |
  | `orig_orientation` | `arma::fmat` | Origin orientations (Euler); `[no_orig, 3]`                                                     |
  | `orig_index`       | `arma::uvec` | Byte offsets from BOF to each origin data block; `[no_orig]`                                    |
  | `path_data_offset` | `arma::uvec` | Absolute offset to path_data_index array per origin; 0 if the TX slot is unwritten; `[no_orig]` |
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
            throw std::invalid_argument("Cannot open file.");
    }
    else
    {
        fileR.seekg(0, std::ios::beg);
        if (!fileR.good())
            throw std::invalid_argument("Supplied ifstream is not in a good state.");
    }

    // Read and validate the header
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

    // Global counters
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

    // CIR metadata (read all)
    if (cache.no_cir != 0)
    {
        uint8_t cir_fmt = 0;
        fileR.read((char *)&cir_fmt, sizeof(uint8_t));

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

    // Skip destination metadata
    if (cache.no_dest != 0)
    {
        fileR.seekg((std::streamoff)(cache.no_dest * sizeof(unsigned)), std::ios::cur);
        for (unsigned i = 0; i < cache.no_dest; ++i)
        {
            uint8_t mt_name_length = 0;
            fileR.read((char *)&mt_name_length, sizeof(uint8_t));
            if (mt_name_length != 0)
                fileR.seekg((std::streamoff)mt_name_length, std::ios::cur);
        }
    }

    // Origin metadata (read all)
    if (cache.no_orig != 0)
    {
        uint8_t orig_fmt = 0;
        fileR.read((char *)&orig_fmt, sizeof(uint8_t));

        cache.orig_pos_all.set_size(cache.no_orig, 3);
        fileR.read((char *)cache.orig_pos_all.memptr(), cache.no_orig * 3 * sizeof(float));

        cache.orig_orientation.zeros(cache.no_orig, 3);
        if (orig_fmt > 3)
            fileR.read((char *)cache.orig_orientation.colptr(0), cache.no_orig * sizeof(float));
        if (orig_fmt == 2 || orig_fmt == 3 || orig_fmt == 6 || orig_fmt == 7)
            fileR.read((char *)cache.orig_orientation.colptr(1), cache.no_orig * sizeof(float));
        if (orig_fmt == 1 || orig_fmt == 3 || orig_fmt == 5 || orig_fmt == 7)
            fileR.read((char *)cache.orig_orientation.colptr(2), cache.no_orig * sizeof(float));

        cache.orig_index.set_size(cache.no_orig);
        fileR.read((char *)cache.orig_index.memptr(), cache.no_orig * sizeof(size_t));
    }

    // Version 4: frequency stored per-origin, read from first origin
    if (cache.version == 4 && cache.no_orig != 0)
    {
        fileR.seekg((std::streampos)cache.orig_index(0));
        uint8_t tx_name_length = 0;
        fileR.read((char *)&tx_name_length, sizeof(uint8_t));
        fileR.seekg((std::streamoff)tx_name_length, std::ios::cur);
        cache.freq.set_size(1);
        fileR.read((char *)cache.freq.memptr(), sizeof(float));
    }

    // Pre-compute path_data_offset for each origin
    // For each origin, this is the absolute byte offset to the start of the
    // path_data_index[] array within that origin's data block.
    // Layout: [tx_name_length(1)] [tx_name(N)] [freq(4, v4 only)] [max_no_path(4)] [path_data_index[no_cir]]
    // So: path_data_offset = orig_index + 1 + tx_name_length + (v4 ? 4 : 0) + 4
    if (cache.no_orig != 0)
    {
        cache.path_data_offset.set_size(cache.no_orig);
        unsigned extra = (cache.version == 4) ? (unsigned)sizeof(float) : 0;

        for (unsigned i = 0; i < cache.no_orig; ++i)
        {
            if (cache.orig_index(i) == 0) // TX not yet written by append
            {
                cache.path_data_offset(i) = 0;
                continue;
            }
            fileR.seekg((std::streampos)cache.orig_index(i));
            uint8_t tx_name_length = 0;
            fileR.read((char *)&tx_name_length, sizeof(uint8_t));
            cache.path_data_offset(i) = cache.orig_index(i) + 1  // tx_name_length byte
                                        + (size_t)tx_name_length // tx_name
                                        + (size_t)extra          // freq (v4 only)
                                        + sizeof(unsigned);      // max_no_path
        }
    }

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
- CIR orientation is stored compressed: only the angle columns that carry a nonzero value are written, encoded in a per-file format byte.
- `no_orig` fixes the number of TX slots; the file can hold at most that many appended TX blocks.
- Positional data is converted to `float` on write regardless of `dtype`.

## Declaration:
```
template <typename dtype>
void quadriga_lib::qrt_file_init(
    const std::string &fn,
    const arma::Col<dtype> &freq,
    const arma::Mat<dtype> &cir_pos,
    const arma::Mat<dtype> &cir_orientation,
    const std::vector<std::string> &dest_names = {"RX"},
    const arma::u32_vec &cir_offset = {0},
    unsigned no_orig = 1,
    bool scalar_mode = false);
```

## Inputs:
- **`fn`** — Path to the QRT file to create (truncated if it exists)
- **`freq`** — Frequencies in GHz (EM) or Hz (scalar); `[n_freq]`, 1 to 127 entries
- **`cir_pos`** — CIR positions in Cartesian coordinates; `[no_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles (bank, tilt, head); `[no_cir, 3]` or empty for none
- **`dest_names`** — Receiver (MT) names; `[no_dest]`
- **`cir_offset`** — CIR offset for each receiver, 0-based; `[no_dest]`, must equal `dest_names` in length
- **`no_orig`** — Number of origin (TX) slots to reserve; at least 1
- **`scalar_mode`** — `true` writes a v6 scalar-layout file, `false` a v5 EM file

## See also:
- [[qrt_file_append]] (write one TX block into a reserved slot)
- [[qrt_file_read]] (read CIR data back)
MD!*/

template <typename dtype>
void quadriga_lib::qrt_file_init(const std::string &fn,
                                 const arma::Col<dtype> &freq,
                                 const arma::Mat<dtype> &cir_pos,
                                 const arma::Mat<dtype> &cir_orientation,
                                 const std::vector<std::string> &dest_names,
                                 const arma::u32_vec &cir_offset,
                                 unsigned no_orig,
                                 bool scalar_mode)
{
    if (freq.n_elem == 0 || freq.n_elem > 127)
        throw std::invalid_argument("qrt_file_init: 'freq' must hold 1..127 frequencies.");
    if (no_orig == 0)
        throw std::invalid_argument("qrt_file_init: 'no_orig' must be at least 1.");

    unsigned no_cir = (unsigned)cir_pos.n_rows;
    if (cir_pos.n_cols != 3)
        throw std::invalid_argument("qrt_file_init: 'cir_pos' must be [n_cir, 3].");
    if (cir_orientation.n_elem != 0 && (cir_orientation.n_rows != no_cir || cir_orientation.n_cols != 3))
        throw std::invalid_argument("qrt_file_init: 'cir_orientation' must be empty or [n_cir, 3].");
    if (dest_names.size() != cir_offset.n_elem)
        throw std::invalid_argument("qrt_file_init: 'dest_names' and 'cir_offset' length mismatch.");

    unsigned no_dest = (unsigned)dest_names.size();
    unsigned no_freq = (unsigned)freq.n_elem;

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
    auto derive_fmt = [](const arma::Mat<dtype> &o, unsigned n) -> uint8_t
    {
        if (o.n_elem == 0 || o.n_rows != n || o.n_cols < 3)
            return 0;
        auto nz = [&](unsigned c)
        { const dtype *p = o.colptr(c); for (unsigned i = 0; i < n; ++i) if (p[i] != (dtype)0) return true; return false; };
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
        throw std::invalid_argument("qrt_file_init: cannot open output file.");

    std::string bin_id = scalar_mode ? "#QRT-BINv06" : "#QRT-BINv05";
    fileW.write(bin_id.c_str(), (std::streamsize)bin_id.size());

    fileW.write((char *)&no_orig, sizeof(unsigned));
    fileW.write((char *)&no_cir, sizeof(unsigned));
    fileW.write((char *)&no_dest, sizeof(unsigned));
    fileW.write((char *)&no_freq, sizeof(unsigned));

    wr_col(fileW, freq.memptr(), no_freq);

    if (no_cir != 0)
    {
        uint8_t cir_fmt = derive_fmt(cir_orientation, no_cir);
        fileW.write((char *)&cir_fmt, sizeof(uint8_t));
        wr_col(fileW, cir_pos.colptr(0), no_cir);
        wr_col(fileW, cir_pos.colptr(1), no_cir);
        wr_col(fileW, cir_pos.colptr(2), no_cir);
        if (cir_fmt & 0x4)
            wr_col(fileW, cir_orientation.colptr(0), no_cir);
        if (cir_fmt & 0x2)
            wr_col(fileW, cir_orientation.colptr(1), no_cir);
        if (cir_fmt & 0x1)
            wr_col(fileW, cir_orientation.colptr(2), no_cir);
    }

    fileW.write((char *)cir_offset.memptr(), no_dest * sizeof(unsigned));
    for (unsigned i = 0; i < no_dest; ++i)
    {
        if (dest_names[i].size() > 255)
            throw std::invalid_argument("qrt_file_init: RX name exceeds 255 characters.");
        uint8_t len = (uint8_t)dest_names[i].size();
        fileW.write((char *)&len, sizeof(uint8_t));
        fileW.write(dest_names[i].data(), (std::streamsize)len);
    }

    uint8_t orig_fmt = 7;
    fileW.write((char *)&orig_fmt, sizeof(uint8_t));
    std::vector<float> zeros_pos(6 * no_orig, 0.0f);
    fileW.write((char *)zeros_pos.data(), (std::streamsize)(zeros_pos.size() * sizeof(float)));

    std::vector<size_t> zero_index(no_orig, 0);
    fileW.write((char *)zero_index.data(), (std::streamsize)(zero_index.size() * sizeof(size_t)));

    fileW.close();
    if (!fileW)
        throw std::runtime_error("qrt_file_init: write error.");
}

template void quadriga_lib::qrt_file_init<float>(const std::string &fn, const arma::Col<float> &freq, const arma::Mat<float> &cir_pos,
                                                 const arma::Mat<float> &cir_orientation, const std::vector<std::string> &dest_names,
                                                 const arma::u32_vec &cir_offset, unsigned no_orig, bool scalar_mode);

template void quadriga_lib::qrt_file_init<double>(const std::string &fn, const arma::Col<double> &freq, const arma::Mat<double> &cir_pos,
                                                  const arma::Mat<double> &cir_orientation, const std::vector<std::string> &dest_names,
                                                  const arma::u32_vec &cir_offset, unsigned no_orig, bool scalar_mode);

/*!MD
# qrt_file_append
Append one transmitter's path data to an existing QRT file

- Writes the path data for a single TX into the next free slot reserved by [[qrt_file_init]], and records the TX name, position, and orientation.
- Paths are grouped by their CIR index (`path::iC`) and stored per CIR as: interaction counts, polarization coefficients, interaction coordinates, and interaction type codes.
- Validates every path before writing: `iC` must be within the file's CIR count, and each path's frequency count and layout mode (EM/scalar) must match the file exactly.
- Throws when all TX slots are already filled — the file holds at most the `no_orig` slots reserved at init.
- Only v5/6 files can be appended to. Positional data is converted to `float` on write.
- Returns the total number of paths written.

## Declaration:
```
template <typename dtype>
size_t quadriga_lib::qrt_file_append(
    const std::string &fn,
    const std::vector<quadriga_lib::path> &path_data,
    const arma::Col<dtype> &orig_pos,
    const arma::Col<dtype> &orig_orientation = {0.0, 0.0, 0.0},
    const std::string &orig_name = "TX");
```

## Inputs:
- **`fn`** — Path to an existing QRT file created by [[qrt_file_init]]
- **`path_data`** — Paths to write; each carries its CIR index in `iC` and must match the file's frequency count and layout mode
- **`orig_pos`** — Transmitter position; `[3]`
- **`orig_orientation`** — Transmitter orientation (bank, tilt, head) in rad; `[3]`
- **`orig_name`** — Transmitter name; at most 255 characters

## Returns:
- Number of paths written across all CIRs.

## See also:
- [[qrt_file_init]] (create the file and reserve TX slots)
- [[qrt_file_read]] (read the appended data back)
MD!*/

template <typename dtype>
size_t quadriga_lib::qrt_file_append(const std::string &fn,
                                     const std::vector<quadriga_lib::path> &path_data,
                                     const arma::Col<dtype> &orig_pos,
                                     const arma::Col<dtype> &orig_orientation,
                                     const std::string &orig_name)
{
    if (orig_pos.n_elem != 3 || orig_orientation.n_elem != 3)
        throw std::invalid_argument("qrt_file_append: 'orig_pos' and 'orig_orientation' must have 3 elements.");
    if (orig_name.size() > 255)
        throw std::invalid_argument("qrt_file_append: 'orig_name' exceeds 255 characters.");

    std::fstream f(fn, std::ios::in | std::ios::out | std::ios::binary);
    if (!f.is_open())
        throw std::invalid_argument("qrt_file_append: cannot open file.");

    // Recover metadata via the existing header parser (open the file read-only for it).
    quadriga_lib::qrt_read_cache cache;
    {
        std::ifstream fr(fn, std::ios::in | std::ios::binary);
        if (!fr.is_open())
            throw std::invalid_argument("qrt_file_append: cannot open file for metadata.");
        cache = quadriga_lib::qrt_read_cache_init(fn, &fr);
    }

    if (cache.version != 5 && cache.version != 6)
        throw std::invalid_argument("qrt_file_append: only v5/6 files can be appended to.");

    unsigned no_cir = cache.no_cir;
    unsigned no_freq = cache.no_freq;
    unsigned no_orig = cache.no_orig;
    bool scalar = (cache.version == 6);
    size_t pol_stride = scalar ? 2 : 8;

    // BS-region offsets. Header up to orig_fmt is fixed-walkable; with orig_fmt = 7 the BS block is
    // orig_fmt(1) + 6 * no_orig floats, followed by the u64 index table.
    // orig_pos_base = start of bs_x[]; bs_index_base = start of bs_data_index[].
    // Derive orig_pos_base by replaying the header length up to and including orig_fmt.
    size_t off = 11 + 4 * sizeof(unsigned); // ID + no_orig/no_cir/no_dest/no_freq
    off += (size_t)no_freq * sizeof(float); // frequencies[]
    if (no_cir != 0)
    {
        // cir_fmt byte + 3 pos columns + orientation columns per cir_fmt.
        // Recover cir_fmt from how many orientation columns are nonzero in the cache.
        uint8_t cir_fmt = 0;
        {
            auto nz = [&](unsigned c)
            { for (unsigned i = 0; i < no_cir; ++i) if (cache.cir_orientation(i, c) != 0.0f) return true; return false; };
            if (nz(0))
                cir_fmt |= 0x4;
            if (nz(1))
                cir_fmt |= 0x2;
            if (nz(2))
                cir_fmt |= 0x1;
        }
        unsigned cols = 3 + ((cir_fmt & 4) ? 1 : 0) + ((cir_fmt & 2) ? 1 : 0) + ((cir_fmt & 1) ? 1 : 0);
        off += 1 + (size_t)cols * no_cir * sizeof(float);
    }
    // MT index + names (variable-length names): walk them from the file.
    {
        f.seekg((std::streamoff)off, std::ios::beg);
        f.seekg((std::streamoff)(cache.no_dest * sizeof(unsigned)), std::ios::cur);
        for (unsigned i = 0; i < cache.no_dest; ++i)
        {
            uint8_t len = 0;
            f.read((char *)&len, sizeof(uint8_t));
            f.seekg((std::streamoff)len, std::ios::cur);
        }
        off = (size_t)f.tellg();
    }
    size_t orig_pos_base = off + 1;                                       // skip orig_fmt byte
    size_t orig_index_base = orig_pos_base + 6 * no_orig * sizeof(float); // after 6 columns

    // Validate paths and bin by CIR (the "2D path index" phase). Throw before any write.
    std::vector<unsigned> no_hit(no_cir, 0);
    for (const auto &p : path_data)
    {
        if ((unsigned)p.iC >= no_cir)
            throw std::out_of_range("qrt_file_append: path CIR index (iC) exceeds no_cir.");
        if ((unsigned)p.n_freq() != no_freq)
            throw std::invalid_argument("qrt_file_append: path frequency count does not match file.");
        if (p.is_scalar() != scalar)
            throw std::invalid_argument("qrt_file_append: path layout mode does not match file.");
        ++no_hit[p.iC];
    }
    unsigned max_no_path = 0;
    for (unsigned c = 0; c < no_cir; ++c)
        if (no_hit[c] > max_no_path)
            max_no_path = no_hit[c];

    std::vector<std::vector<const quadriga_lib::path *>> by_cir(no_cir);
    for (unsigned c = 0; c < no_cir; ++c)
        by_cir[c].reserve(no_hit[c]);
    for (const auto &p : path_data)
        by_cir[p.iC].push_back(&p);

    // First free BS slot (0 = free); throw if full.
    unsigned slot = no_orig;
    f.seekg((std::streamoff)orig_index_base, std::ios::beg);
    for (unsigned i = 0; i < no_orig; ++i)
    {
        size_t v = 0;
        f.read((char *)&v, sizeof(size_t));
        if (v == 0)
        {
            slot = i;
            break;
        }
    }
    if (slot == no_orig)
        throw std::runtime_error("qrt_file_append: BS index table is full; no free slot.");

    // Patch this TX's position/orientation row (columns x,y,z,bank,tilt,head; orig_fmt = 7).
    auto patch_col = [&](unsigned col, float v)
    {
        size_t o = orig_pos_base + size_t(col * no_orig + slot) * sizeof(float);
        f.seekp((std::streamoff)o, std::ios::beg);
        f.write((char *)&v, sizeof(float));
    };
    patch_col(0, (float)orig_pos(0));
    patch_col(1, (float)orig_pos(1));
    patch_col(2, (float)orig_pos(2));
    patch_col(3, (float)orig_orientation(0));
    patch_col(4, (float)orig_orientation(1));
    patch_col(5, (float)orig_orientation(2));

    // Append the bs_data_t block at EOF.
    f.seekp(0, std::ios::end);
    size_t block_off = (size_t)f.tellp();

    uint8_t name_len = (uint8_t)orig_name.size();
    f.write((char *)&name_len, sizeof(uint8_t));
    f.write(orig_name.data(), (std::streamsize)name_len);
    f.write((char *)&max_no_path, sizeof(unsigned));

    std::streamoff index_pos = (std::streamoff)f.tellp();
    std::vector<size_t> path_index(no_cir, 0);
    f.write((char *)path_index.data(), (std::streamsize)(path_index.size() * sizeof(size_t)));

    size_t total_paths = 0;
    for (unsigned c = 0; c < no_cir; ++c)
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
            std::vector<float> pol(pol_stride * nP * no_freq);
            for (unsigned i_freq = 0; i_freq < no_freq; ++i_freq)
                for (unsigned iP = 0; iP < nP; ++iP)
                {
                    const float *cf = paths[iP]->xpr_coeff(i_freq);
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
        throw std::runtime_error("qrt_file_append: write error.");
    return total_paths;
}

template size_t quadriga_lib::qrt_file_append<float>(const std::string &fn, const std::vector<quadriga_lib::path> &path_data,
                                                     const arma::Col<float> &tx_pos, const arma::Col<float> &tx_orientation,
                                                     const std::string &tx_name);

template size_t quadriga_lib::qrt_file_append<double>(const std::string &fn, const std::vector<quadriga_lib::path> &path_data,
                                                      const arma::Col<double> &tx_pos, const arma::Col<double> &tx_orientation,
                                                      const std::string &tx_name);

/*!MD
# qrt_file_parse
Read metadata from a QRT file

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count, CIR offsets, names, positions, orientations, and file version.
- All output arguments are optional; pass `nullptr` to skip any.
- If `file` is `nullptr`, the file is opened internally and closed on return; if provided, the stream is left open.
- When `no_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and `cir_offset` reflect this.

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
    int *version = nullptr,
    arma::fvec *freq = nullptr,
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
- **`no_cir`** *(optional)* — Number of channel snapshots per origin point
- **`no_orig`** *(optional)* — Number of origin points (TX)
- **`no_dest`** *(optional)* — Number of destination points (RX)
- **`no_freq`** *(optional)* — Number of frequency bands
- **`cir_offset`** *(optional)* — CIR offset per destination; `[no_dest]`
- **`orig_names`** *(optional)* — Names of origin points; `[no_orig]`
- **`dest_names`** *(optional)* — Names of destination points; `[no_dest]`
- **`version`** *(optional)* — QRT file version number
- **`freq`** *(optional)* — Frequencies as stored in the file; usually in GHz; `[no_freq]`
- **`cir_pos`** *(optional)* — CIR positions in Cartesian coordinates; `[no_cir, 3]`
- **`cir_orientation`** *(optional)* — CIR orientations as Euler angles; `[no_cir, 3]`
- **`orig_pos`** *(optional)* — Origin (TX) positions in Cartesian coordinates; `[no_orig, 3]`
- **`orig_orientation`** *(optional)* — Origin (TX) orientations as Euler angles; `[no_orig, 3]`
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
                                  arma::fvec *freq,
                                  arma::fmat *cir_pos,
                                  arma::fmat *cir_orientation,
                                  arma::fmat *orig_pos,
                                  arma::fmat *orig_orientation,
                                  std::ifstream *file)
{
    // Determine which stream to use and whether we own it
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

    // Read and validate the header
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

    // Global counters
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

    // CIR metadata
    arma::fmat l_cir_pos;
    arma::fmat l_cir_orientation;
    uint8_t cir_fmt = 0;

    if (l_no_cir != 0)
    {
        fileR.read((char *)&cir_fmt, sizeof(uint8_t));

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
        l_cir_index.set_size(l_no_dest);
        fileR.read((char *)l_cir_index.memptr(), l_no_dest * sizeof(unsigned));

        l_dest_names.resize((size_t)l_no_dest);
        for (unsigned i = 0; i < l_no_dest; ++i)
        {
            uint8_t mt_name_length = 0;
            fileR.read((char *)&mt_name_length, sizeof(uint8_t));
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

    // Origin (TX) metadata
    arma::fmat l_orig_pos_all;
    arma::fmat l_orig_orientation;
    arma::uvec l_orig_index;

    if (l_no_orig != 0)
    {
        uint8_t orig_fmt = 0;
        fileR.read((char *)&orig_fmt, sizeof(uint8_t));

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

    // Populate output parameters
    if (no_cir)
        *no_cir = (arma::uword)l_no_cir;
    if (no_orig)
        *no_orig = (arma::uword)l_no_orig;
    if (no_dest)
        *no_dest = (arma::uword)(l_no_dest == 0 ? 1 : l_no_dest);
    if (no_freq)
        *no_freq = (arma::uword)l_no_freq;
    if (version)
        *version = ver;

    if (l_cir_index.n_elem == 0 || l_cir_index[0] != 0)
        throw std::invalid_argument("Invalid CIR index in QRT file. Potential file corruption.");

    if (cir_offset)
    {
        cir_offset->set_size(l_cir_index.n_elem);
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
            if (l_orig_index(i) == 0) // TX slot reserved but not yet written by append
            {
                orig_names->push_back(std::string());
                continue;
            }
            fileR.seekg((std::streampos)l_orig_index(i));

            uint8_t tx_name_length = 0;
            fileR.read((char *)&tx_name_length, sizeof(uint8_t));

            std::string name((size_t)tx_name_length, '\0');
            fileR.read(&name[0], (size_t)tx_name_length);

            orig_names->push_back(std::move(name));
        }
    }

    if (freq)
    {
        if (ver == 4 && l_no_orig != 0)
        {
            // In version 4 the frequency is stored per-origin; read from the first origin
            fileR.seekg((std::streampos)l_orig_index(0));

            uint8_t tx_name_length = 0;
            fileR.read((char *)&tx_name_length, sizeof(uint8_t));
            fileR.seekg((std::streamoff)tx_name_length, std::ios::cur); // skip name

            l_freq.set_size(1);
            fileR.read((char *)l_freq.memptr(), sizeof(float));
        }
        *freq = std::move(l_freq);
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

- Reads channel impulse response data for a specific snapshot index and origin point.
- All output arguments are optional; pass `nullptr` to skip any.
- If `downlink = true`, origin is TX and destination is RX; if `false`, roles are swapped.
- For tight-loop performance, pass a pre-opened `std::ifstream` and a [[qrt_read_cache_init]]-populated cache; reduces per-call I/O to 2 seeks and 4 reads.
- `fn` is ignored when both `file` and `cache` are provided.
- Reading a TX slot that was reserved by [[qrt_file_init]] but not yet written by [[qrt_file_append]] returns
  empty path outputs with zeroed positions, rather than throwing; an out-of-range index still throws.

## Declaration:
```
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
    std::vector<uint8_t> *interact_type = nullptr,
    std::ifstream *file = nullptr,
    const qrt_read_cache *cache = nullptr);
```

## Inputs:
- **`fn`** — Path to the QRT file; ignored when both `file` and `cache` are provided
- **`i_cir`** — Snapshot index, 0-based
- **`i_orig`** — Origin index, 0-based
- **`downlink`** — If `true`, origin=TX, destination=RX; if `false`, roles are swapped
- **`normalize_M`** *(optional)* — Controls `M` and `path_gain` scaling where PL is the propagation-only path loss
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm<br><br>
  | `normalize_M` | `M`                   | `path_gain`                      |
  | ------------- | --------------------- | -------------------------------- |
  | 0             | As stored in QRT file | -PL                              |
  | 1             | Max column power = 1  | -PL minus material losses        |
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; left open on return
- **`cache`** *(optional)* — Pre-populated cache from [[qrt_read_cache_init]]<br><br>

## Outputs:
- **`center_frequency`** — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3]`
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Path gain on linear scale; `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; `[n_path]`
- **`M`** — Polarization transfer matrix; `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; `[n_path]`
- **`eod`** — Departure elevation angles; `[n_path]`
- **`aoa`** — Arrival azimuth angles; `[n_path]`
- **`eoa`** — Arrival elevation angles; `[n_path]`
- **`path_coord`** — Interaction coordinates per path; vector of length `n_path`, each `[3, n_interact + 2]`
- **`no_int`** — Number of mesh interactions per path; 0 indicates LOS; `[n_path]`
- **`coord`** — Interaction coordinates; `[3, sum(no_int)]`
- **`interact_type`** — Interaction type codes, concatenated per path and segmented by `no_int`; `[sum(no_int)]`.
  Empty for v4 legacy files.

## Example:
```
std::ifstream stream("scene.qrt", std::ios::in | std::ios::binary);
auto cache = quadriga_lib::qrt_read_cache_init("scene.qrt", &stream);
arma::vec center_freq, tx_pos, rx_pos, path_length;
arma::mat path_gain; arma::cube M;
for (arma::uword ic = 0; ic < cache.no_cir; ++ic)
    for (arma::uword io = 0; io < cache.no_orig; ++io)
        quadriga_lib::qrt_file_read<double>("", ic, io, true,
            &center_freq, &tx_pos, nullptr, &rx_pos, nullptr,
            nullptr, nullptr, &path_gain, &path_length, &M,
            nullptr, nullptr, nullptr, nullptr, nullptr, 1,
            nullptr, nullptr, &stream, &cache);
```

## See also:
- [[qrt_read_cache_init]] (populate cache for fast repeated reads)
- [[qrt_file_parse]] (extract file metadata without reading CIR data)
MD!*/

template <typename dtype>
void quadriga_lib::qrt_file_read(const std::string &fn, arma::uword i_cir, arma::uword i_orig, bool downlink,
                                 arma::Col<dtype> *center_frequency, arma::Col<dtype> *tx_pos, arma::Col<dtype> *tx_orientation,
                                 arma::Col<dtype> *rx_pos, arma::Col<dtype> *rx_orientation,
                                 arma::Mat<dtype> *fbs_pos, arma::Mat<dtype> *lbs_pos,
                                 arma::Mat<dtype> *path_gain, arma::Col<dtype> *path_length, arma::Cube<dtype> *M,
                                 arma::Col<dtype> *aod, arma::Col<dtype> *eod, arma::Col<dtype> *aoa, arma::Col<dtype> *eoa,
                                 std::vector<arma::Mat<dtype>> *path_coord, int normalize_M, arma::u32_vec *no_int, arma::fmat *coord, std::vector<uint8_t> *interact_type,
                                 std::ifstream *file, const qrt_read_cache *cache)
{
    // Stream setup
    std::ifstream local_stream;
    bool own_stream = (file == nullptr);
    std::ifstream &fileR = own_stream ? local_stream : *file;

    if (own_stream)
    {
        fileR.open(fn, std::ios::in | std::ios::binary);
        if (!fileR.is_open())
            throw std::invalid_argument("Cannot open file.");
    }

    // Obtain metadata
    int ver;
    unsigned l_no_cir, l_no_freq;
    const float *p_freq;                                  // points at freq data (no copy needed)
    float cir_px, cir_py, cir_pz;                         // CIR position
    float cir_ox = 0.0f, cir_oy = 0.0f, cir_oz = 0.0f;    // CIR orientation
    float orig_px, orig_py, orig_pz;                      // Origin position
    float orig_ox = 0.0f, orig_oy = 0.0f, orig_oz = 0.0f; // Origin orientation
    size_t l_path_data_offset;                            // Absolute byte offset to path_data_index array

    arma::fvec l_freq_buf; // Local buffer for slow path; must outlive p_freq usage

    if (cache) // FAST PATH: metadata from cache (zero I/O)
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
    else // SLOW PATH: parse header + selective metadata from file =======
    {
        fileR.seekg(0, std::ios::beg);
        if (!fileR.good())
            throw std::invalid_argument("Supplied ifstream is not in a good state.");

        // Read and validate the header
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

        // Global counters
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

        // Frequencies (v5/v6: stored in header)
        if (ver > 4)
        {
            fileR.read((char *)&l_no_freq, sizeof(unsigned));
            l_freq_buf.set_size(l_no_freq);
            fileR.read((char *)l_freq_buf.memptr(), l_no_freq * sizeof(float));
        }

        // CIR metadata (selective read for i_cir)
        {
            uint8_t cir_fmt = 0;
            fileR.read((char *)&cir_fmt, sizeof(uint8_t));

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

        // Skip destination metadata
        if (l_no_dest != 0)
        {
            fileR.seekg((std::streamoff)(l_no_dest * sizeof(unsigned)), std::ios::cur);
            for (unsigned i = 0; i < l_no_dest; ++i)
            {
                uint8_t mt_name_length = 0;
                fileR.read((char *)&mt_name_length, sizeof(uint8_t));
                if (mt_name_length != 0)
                    fileR.seekg((std::streamoff)mt_name_length, std::ios::cur);
            }
        }

        // Origin metadata (selective read for i_orig)
        {
            uint8_t orig_fmt = 0;
            fileR.read((char *)&orig_fmt, sizeof(uint8_t));

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

            if (orig_fmt > 3) // Bank angle
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                orig_ox = val;
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (orig_fmt == 2 || orig_fmt == 3 || orig_fmt == 6 || orig_fmt == 7) // Tilt angle
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                orig_oy = val;
                fileR.seekg(skip_after, std::ios::cur);
            }
            if (orig_fmt == 1 || orig_fmt == 3 || orig_fmt == 5 || orig_fmt == 7) // Heading angle
            {
                fileR.seekg(skip_before, std::ios::cur);
                fileR.read((char *)&val, sizeof(float));
                orig_oz = val;
                fileR.seekg(skip_after, std::ios::cur);
            }

            // Read orig_index[i_orig] and compute path_data_offset
            std::streamoff skip64_before = std::streamoff(i_orig * sizeof(size_t));
            fileR.seekg(skip64_before, std::ios::cur);

            size_t l_orig_index;
            fileR.read((char *)&l_orig_index, sizeof(size_t));

            // Seek to the origin block, read tx_name_length, compute offset
            // orig_index == 0 marks a reserved-but-unwritten TX slot (see qrt_file_init/append).
            if (l_orig_index == 0)
                l_path_data_offset = 0;
            else
            {
                fileR.seekg((std::streampos)l_orig_index);
                uint8_t tx_name_length = 0;
                fileR.read((char *)&tx_name_length, sizeof(uint8_t));
                l_path_data_offset = l_orig_index + 1 + (size_t)tx_name_length + (size_t)((ver == 4) ? sizeof(float) : 0) + sizeof(unsigned);
            }
        }

        // Version 4: frequency stored per-origin
        if (ver == 4 && l_path_data_offset != 0)
        {
            size_t freq_offset = l_path_data_offset - sizeof(unsigned) - sizeof(float);
            fileR.seekg((std::streampos)freq_offset);
            l_freq_buf.set_size(1);
            fileR.read((char *)l_freq_buf.memptr(), sizeof(float));
        }
        else if (ver == 4)
        {
            l_freq_buf.set_size(1);
            l_freq_buf(0) = 0.0f; // unfilled v4 slot: no per-origin freq available
        }

        p_freq = l_freq_buf.memptr();
    }

    arma::uword no_freq = (arma::uword)l_no_freq;
    bool v6 = (ver == 6);

    // TX slot reserved but not yet appended
    if (l_path_data_offset == 0)
    {
        // Positions/orientations already default to the reserved zeros; report no paths.
        if (center_frequency)
        {
            center_frequency->set_size(no_freq);
            for (arma::uword k = 0; k < no_freq; ++k)
                center_frequency->at(k) = v6 ? (dtype)p_freq[k] : (dtype)p_freq[k] * (dtype)1e9;
        }
        if (tx_pos)
            tx_pos->zeros(3);
        if (rx_pos)
            rx_pos->zeros(3);
        if (tx_orientation)
            tx_orientation->zeros(3);
        if (rx_orientation)
            rx_orientation->zeros(3);
        if (no_int)
            no_int->reset();
        if (coord)
            coord->reset();
        if (interact_type)
            interact_type->clear();
        if (M)
            M->reset();
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
        if (own_stream && fileR.is_open())
            fileR.close();
        return;
    }

    // Read CIR path data (2 seeks + 4 reads)
    // Seek directly to path_data_index[i_cir] within the origin block
    fileR.seekg((std::streampos)(l_path_data_offset + i_cir * sizeof(size_t)));

    size_t data_offset = 0;
    fileR.read((char *)&data_offset, sizeof(size_t));

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
            uint8_t no_mesh_interact_byte = 0;
            fileR.read((char *)&no_mesh_interact_byte, sizeof(uint8_t));
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

    // Interaction type codes (v5/v6 only; absent in v4 legacy files)
    std::vector<uint8_t> interact_typeR;
    if (ver != 4)
    {
        interact_typeR.resize(sum_no_int);
        uint8_t *p_it = interact_typeR.data();
        for (unsigned k = 0; k < sum_no_int; ++k)
        {
            uint8_t b = 0;
            fileR.read((char *)&b, sizeof(uint8_t));
            p_it[k] = b;
        }
    }

    // Close only if we opened the stream ourselves
    if (own_stream && fileR.is_open())
        fileR.close();

    // Gain at 1 m (path-loss reference)
    arma::vec gain_at_1m(no_freq);
    if (v6) // Scalar: 1/r spreading, frequency-independent reference at 1 m.
        gain_at_1m.zeros();
    else
        for (arma::uword i_freq = 0; i_freq < no_freq; ++i_freq)
            gain_at_1m[i_freq] = -32.45 - 20.0 * std::log10((double)p_freq[i_freq]);

    // Air absorption α(f) in dB/m (ISO 9613-1)
    // Defaults: T = 20 °C, RH = 50 %, p = 1 atm (indoor sea-level).
    // Zero for v4/v5 (EM) so the per-path loop works uniformly.
    arma::vec alpha_dB_per_m(no_freq, arma::fill::zeros);
    if (v6)
    {
        auto iso9613_alpha = [](double f_Hz,
                                double T_celsius = 20.0,
                                double RH_percent = 50.0,
                                double p_kPa = 101.325) -> double
        {
            constexpr double T0 = 293.15;   // Reference temperature [K]
            constexpr double T01 = 273.16;  // Triple-point isotherm [K]
            constexpr double p_r = 101.325; // Reference pressure [kPa]

            const double T = T_celsius + 273.15; // Absolute temperature [K]

            // Saturation vapor pressure ratio and molar concentration of water vapor [%]
            const double C = -6.8346 * std::pow(T01 / T, 1.261) + 4.6151;
            const double psat = std::pow(10.0, C);
            const double h = RH_percent * psat / (p_kPa / p_r);

            // Relaxation frequencies for O2 and N2 [Hz]
            const double pa_pr = p_kPa / p_r;
            const double T_T0 = T / T0;
            const double f_rO = pa_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
            const double f_rN = pa_pr * std::pow(T_T0, -0.5) * (9.0 + 280.0 * h * std::exp(-4.170 * (std::pow(T_T0, -1.0 / 3.0) - 1.0)));

            // Absorption coefficient [dB/m]
            const double f2 = f_Hz * f_Hz;
            const double classical = 1.84e-11 / pa_pr * std::sqrt(T_T0);
            const double rot_O = 0.01275 * std::exp(-2239.1 / T) / (f_rO + f2 / f_rO);
            const double rot_N = 0.1068 * std::exp(-3352.0 / T) / (f_rN + f2 / f_rN);

            return 8.686 * f2 * (classical + std::pow(T_T0, -2.5) * (rot_O + rot_N));
        };

        // p_freq is stored in Hz (see center_frequency export)
        for (arma::uword i_freq = 0; i_freq < no_freq; ++i_freq)
            alpha_dB_per_m[i_freq] = iso9613_alpha((double)p_freq[i_freq]);
    }

    // Output: center_frequency
    if (center_frequency)
    {
        center_frequency->set_size(no_freq);
        for (arma::uword i_freq = 0; i_freq < no_freq; ++i_freq)
            center_frequency->at(i_freq) = v6 ? (dtype)p_freq[i_freq] : (dtype)p_freq[i_freq] * (dtype)1e9;
    }

    // Output: positions
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

    // Output: orientations
    if (tx_orientation && downlink)
        *tx_orientation = {(dtype)orig_ox, (dtype)orig_oy, (dtype)orig_oz};
    if (tx_orientation && !downlink)
        *tx_orientation = {(dtype)cir_ox, (dtype)cir_oy, (dtype)cir_oz};

    if (rx_orientation && downlink)
        *rx_orientation = {(dtype)cir_ox, (dtype)cir_oy, (dtype)cir_oz};
    if (rx_orientation && !downlink)
        *rx_orientation = {(dtype)orig_ox, (dtype)orig_oy, (dtype)orig_oz};

    // Output: raw interaction data
    if (no_int)
        *no_int = no_intR;

    if (coord)
        *coord = coordR;

    if (interact_type)
        *interact_type = interact_typeR;

    // Calculate path gain and polarization matrix M
    // - xprmatR includes all interaction losses, but not the path loss
    // - here we calculate the normalized polarization matrix M and the PG without path loss
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

        // Adjust path gain to include the path loss
        if (want_length)
        {
            if (path_length)
                path_length->set_size(no_path);

            dtype *src = path_length_local.memptr();
            dtype *pg = path_gain ? path_gain->memptr() : nullptr;
            dtype *pl = path_length ? path_length->memptr() : nullptr;
            double *p_gain_at_1m = gain_at_1m.memptr();
            double *p_alpha = alpha_dB_per_m.memptr();

            for (arma::uword i_path = 0; i_path < no_path; ++i_path)
            {
                dtype len = src[i_path];
                if (pl)
                    pl[i_path] = len;
                if (pg)
                {
                    for (arma::uword i_freq = 0; i_freq < no_freq; ++i_freq)
                    {
                        double gainPL = p_gain_at_1m[i_freq] - 20.0 * std::log10((double)len) - p_alpha[i_freq] * (double)len;
                        pg[i_freq * no_path + i_path] *= (dtype)std::pow(10.0, 0.1 * gainPL);
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
}

template void quadriga_lib::qrt_file_read(const std::string &fn, arma::uword i_cir, arma::uword i_orig, bool downlink,
                                          arma::Col<float> *center_frequency, arma::Col<float> *tx_pos, arma::Col<float> *tx_orientation,
                                          arma::Col<float> *rx_pos, arma::Col<float> *rx_orientation,
                                          arma::Mat<float> *fbs_pos, arma::Mat<float> *lbs_pos,
                                          arma::Mat<float> *path_gain, arma::Col<float> *path_length, arma::Cube<float> *M,
                                          arma::Col<float> *aod, arma::Col<float> *eod, arma::Col<float> *aoa, arma::Col<float> *eoa,
                                          std::vector<arma::Mat<float>> *path_coord, int normalize_M, arma::u32_vec *no_int, arma::fmat *coord,
                                          std::vector<uint8_t> *interact_type, std::ifstream *file, const qrt_read_cache *cache);

template void quadriga_lib::qrt_file_read(const std::string &fn, arma::uword i_cir, arma::uword i_orig, bool downlink,
                                          arma::Col<double> *center_frequency, arma::Col<double> *tx_pos, arma::Col<double> *tx_orientation,
                                          arma::Col<double> *rx_pos, arma::Col<double> *rx_orientation,
                                          arma::Mat<double> *fbs_pos, arma::Mat<double> *lbs_pos,
                                          arma::Mat<double> *path_gain, arma::Col<double> *path_length, arma::Cube<double> *M,
                                          arma::Col<double> *aod, arma::Col<double> *eod, arma::Col<double> *aoa, arma::Col<double> *eoa,
                                          std::vector<arma::Mat<double>> *path_coord, int normalize_M, arma::u32_vec *no_int, arma::fmat *coord,
                                          std::vector<uint8_t> *interact_type, std::ifstream *file, const qrt_read_cache *cache);

/*!MD
# qrt_file_read_raw
Read raw ray-tracing path data from a QRT file into path objects

- Reassembles the stored path data for a single origin (TX) into a vector of [[path]] objects — the inverse of [[qrt_file_append]].
- Unlike [[qrt_file_read]], which returns processed CIR data (channel matrices, angles, path loss), this returns the raw per-path storage:
  coordinates, polarization coefficients, and interaction type codes.
- Each returned path has `iC` set to its CIR index and `iR` to a running index within the origin. Counters not stored in the file
  (`nREF`, `nTRA`, `nSUB`, `nSCT`) and `length` are left at their defaults.
- Only v05/v06 files can be read as raw paths; v4 legacy files are rejected.
- An origin slot that was reserved by [[qrt_file_init]] but never written by [[qrt_file_append]] returns an empty vector.
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
- **`i_orig`** — Origin index to read (for downlink, origin = TX); must be less than `no_orig`
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally
- **`cache`** *(optional)* — Pre-parsed metadata from [[qrt_read_cache_init]]; pass `nullptr` to parse the header on this call

## Returns:
- Vector of [[path]] objects for the requested origin, one per stored path across all CIRs; empty if the origin slot is unwritten.

## See also:
- [[qrt_file_append]] (write path data — the inverse operation)
- [[qrt_read_cache_init]] (populate cache for fast repeated reads)
- [[qrt_file_read]] (read processed CIR data instead of raw paths)
MD!*/

// Read raw ray-tracing data from QRT file
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
            throw std::invalid_argument("qrt_file_read_raw: cannot open file.");
    }
    else if (!fileR.good())
        throw std::invalid_argument("qrt_file_read_raw: supplied ifstream is not in a good state.");

    // Metadata: use the supplied cache, or build a local one.
    quadriga_lib::qrt_read_cache local_cache;
    const quadriga_lib::qrt_read_cache *c = cache;
    if (c == nullptr)
    {
        local_cache = quadriga_lib::qrt_read_cache_init(fn, &fileR);
        c = &local_cache;
    }

    if (c->version != 5 && c->version != 6)
        throw std::invalid_argument("qrt_file_read_raw: only v5/6 files can be read as raw paths.");
    if ((unsigned)i_orig >= c->no_orig)
        throw std::out_of_range("qrt_file_read_raw: origin index exceeds no_orig.");

    unsigned no_cir = c->no_cir;
    unsigned no_freq = c->no_freq;
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

    // Per-CIR path_data_index for this origin (no_cir u64 offsets).
    std::vector<size_t> cir_offset(no_cir);
    fileR.seekg((std::streampos)pdo, std::ios::beg);
    fileR.read((char *)cir_offset.data(), (std::streamsize)(no_cir * sizeof(size_t)));

    for (unsigned ci = 0; ci < no_cir; ++ci)
    {
        fileR.seekg((std::streampos)cir_offset[ci], std::ios::beg);

        unsigned nP = 0;
        fileR.read((char *)&nP, sizeof(unsigned));
        if (nP == 0)
            continue;

        // no_mesh_interact[nP] = n_seg() per path
        std::vector<uint8_t> nmi(nP);
        fileR.read((char *)nmi.data(), (std::streamsize)nP);
        size_t sum_int = 0;
        for (unsigned i = 0; i < nP; ++i)
            sum_int += nmi[i];

        // polarization[stride * nP * no_freq], freq-major then path
        std::vector<float> pol(stride * nP * no_freq);
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
            throw std::runtime_error("qrt_file_read_raw: unexpected end of file in path block.");

        // Reassemble one path object per stored path.
        size_t coord_cursor = 0, type_cursor = 0;
        out.reserve(out.size() + nP);
        for (unsigned iP = 0; iP < nP; ++iP)
        {
            unsigned ns = nmi[iP];

            quadriga_lib::path p(ns, no_freq, scalar); // allocates buffer, seeds identity
            p.iC = ci;

            // Coordinates -> coord buffer (3 * ns floats)
            if (ns != 0)
            {
                std::memcpy(p.coord(0), coords.data() + coord_cursor * 3, (size_t)ns * 3 * sizeof(float));
                coord_cursor += ns;
            }

            // Polarization -> xpr_coeff slots, undoing the freq-major interleave
            for (unsigned i_freq = 0; i_freq < no_freq; ++i_freq)
            {
                const float *src = pol.data() + ((size_t)i_freq * nP + iP) * stride;
                std::memcpy(p.xpr_coeff(i_freq), src, stride * sizeof(float));
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
