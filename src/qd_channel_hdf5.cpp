// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "qd_hdf5.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace h5 = qd::hdf5;

/*!SECTION
Channel functions
SECTION!*/

// Internal helpers
// - Channel files are HDF5 files of the format version 3 (dataset 'Version' = 3), written and read with the HDF5
//   subset of qd_hdf5.hpp. This file holds the schema: the root datasets 'ChannelDims' (layout {nx, ny, nz, nw}),
//   'Order' (channel ID per slot, 0 = empty) and 'ChannelRef' (object reference per slot to the channel group, 0 =
//   empty), one group 'channel_<id>' per written slot with ten attributes, the datasets of the structured data
//   (directly in the channel group for one snapshot, in groups 'Snap_<i>' otherwise), and the group 'Data' for the
//   unstructured datasets in creation order.
// - Readers locate a slot through 'ChannelRef' only. Writers append all new objects of a call as one block at the
//   end-of-file address and then update the file in place, in this order: superblock (new end of file), the root
//   link of the new channel group, 'ChannelRef' and 'Order' of the slot, and (overwrite) the removal of the old link.
//   Every intermediate state is a valid file. All inputs and all file structures a writer needs are checked before
//   the first byte is written.
// - Malformed files give std::invalid_argument, failing writes std::runtime_error.

// Element type of a numeric unstructured data type (type IDs of quadriga_lib::any_type_id); false if unsupported
static bool qHDF_par_type(int datatype_id, h5::dtype &t)
{
    if (datatype_id < 10 || datatype_id > 55 || datatype_id % 10 > 5)
        return false;
    const h5::dtype types[6] = {h5::float32, h5::float64, h5::uint64, h5::int64, h5::uint32, h5::int32};
    t = types[datatype_id % 10];
    return true;
}

// Check the name (including the prefix) and the type of an unstructured dataset before anything is written
static void qHDF_check_par(const std::string &name, const std::any *par_data)
{
    auto is_alnum = [&]()
    {
        for (char ch : name)
            if (!std::isalnum((unsigned char)ch) && ch != '_')
                return false;
        return true;
    };
    if (name.length() == 0 || !is_alnum())
        throw std::invalid_argument("Parameter name must only contain letters, numbers and the underscore '_'.");
    if (name.length() > h5::max_name_length)
        throw std::invalid_argument("Parameter name is too long.");
    if (par_data == nullptr)
        throw std::invalid_argument("NULL pointer was passed as data object.");
    h5::dtype t = h5::float32;
    const int datatype_id = quadriga_lib::any_type_id(par_data);
    if (datatype_id != 9 && !qHDF_par_type(datatype_id, t))
        throw std::invalid_argument("Unsupported data type for '" + name + "'");
    if (datatype_id == 9 && std::any_cast<std::string>(par_data)->size() > (size_t)UINT32_MAX) // 32-bit type size
        throw std::invalid_argument("String '" + name + "' is too long: at most 4294967295 bytes can be stored.");
}

// Number of slots of a layout, computed in 64 bits; throws if it exceeds 2^32 - 1
static unsigned long long qHDF_slot_count(unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    unsigned long long n = 1ULL;
    for (unsigned d : {nx, ny, nz, nw})
        if ((n *= (unsigned long long)d) > (unsigned long long)UINT_MAX)
            throw std::invalid_argument("Storage layout has too many slots: nx * ny * nz * nw must not exceed 4294967295.");
    return n;
}

// An open channel file
struct qHDF_file
{
    h5::file f;
    h5::chunk root;                                       // Root group, chunk 0
    uint64_t eof = 0;                                     // End-of-file address of the superblock
    unsigned dims[4] = {0, 0, 0, 0};                      // Storage layout {nx, ny, nz, nw}
    uint64_t n_slot = 0;                                  // nx * ny * nz * nw
    uint64_t dims_addr = 0, order_addr = 0, ref_addr = 0; // Data of 'ChannelDims', 'Order', 'ChannelRef'
    std::vector<std::pair<uint64_t, uint64_t>> index;     // Address and length of these three data ranges
    uint64_t bytes_read = 0;                              // Header and raw bytes read for the channel; at most 'eof'
};

// True if two address ranges {address, length} overlap
static inline bool qHDF_overlap(const std::pair<uint64_t, uint64_t> &a, const std::pair<uint64_t, uint64_t> &b)
{
    return a.first < b.first + b.second && b.first < a.first + a.second; // Ranges lie within the file: no wrap
}

// Writers: the chunks a writer may rewrite must not overlap the index data, which it also writes in place
static void qHDF_check_chunks(const qHDF_file &h, const std::vector<std::pair<uint64_t, uint64_t>> &chunks)
{
    for (const auto &c : chunks)
        for (const auto &r : h.index)
            if (qHDF_overlap(c, r))
                h5::malformed("the storage index overlaps the object header at address " + std::to_string(c.first));
}

// Open a channel file: superblock, root chunk 0 with the links 'ChannelDims', 'Order', 'Version', 'ChannelRef' (each
// exactly once), the version, and the storage index. With 'update', the file is opened for writing and the root
// chunk must have exactly the structure written by the library (other links in the root would be lost). Returns
// false if the file does not exist.
static bool qHDF_open(qHDF_file &h, const std::string &fn, bool update, const char *open_error = "Error opening file.")
{
    if (!h.f.open(fn, update ? "r+b" : "rb"))
    {
        if (!h5::file::exists(fn))
            return false;
        throw std::invalid_argument(open_error);
    }
    const h5::superblock sb = h5::read_superblock(h.f);
    h.eof = sb.eof;
    h.root = h5::read_chunk(h.f, sb.root, 0, h.eof);

    const char *const names[4] = {"ChannelDims", "Order", "Version", "ChannelRef"};
    uint64_t target[4] = {0, 0, 0, 0};
    int count[4] = {0, 0, 0, 0};
    size_t position[4] = {0, 0, 0, 0}; // Message index of each link
    for (size_t i = 0; i < h.root.messages.size(); ++i)
    {
        const h5::message_ref &m = h.root.messages[i];
        uint8_t flags = 0;
        uint64_t max_corder = 0;
        h5::link l;
        if (m.type == h5::msg_link_info && !h5::decode_link_info(h.root.data(m), m.size, flags, max_corder))
            h5::malformed("invalid link info message in the root group");
        if (m.type == h5::msg_link && !h5::decode_link(h.root.data(m), m.size, l))
            h5::malformed("invalid link message in the root group");
        for (int k = 0; k < 4 && m.type == h5::msg_link; ++k)
            if (l.name == names[k])
                ++count[k], target[k] = l.target, position[k] = i;
    }
    for (int k = 0; k < 4; ++k)
        if (count[k] != 1)
            h5::unsupported();
    if (update) // Link Info, Group Info, then the four links in this order
    {
        h5::root_links::check_chunk0(h.root, 4);
        for (int k = 0; k < 4; ++k)
            if (position[k] != 2 + (size_t)k)
                h5::unsupported();
    }

    // Root chunk 0, the root dataset headers and the index data must not overlap: the index data is written in place
    std::vector<std::pair<uint64_t, uint64_t>> regions = {{h.root.addr, h.root.length()}};
    h5::object o;
    auto root_dataset = [&](int k)
    {
        o = h5::read_object(h.f, target[k], h.eof);
        const auto r = h5::chunk_ranges(o);
        regions.insert(regions.end(), r.begin(), r.end());
        return h5::decode_dataset(o, h.eof);
    };

    h5::dataset d = root_dataset(2);
    if (d.type != h5::uint32 || d.rank != 0)
        h5::malformed("dataset 'Version' must be a scalar of type uint32");
    uint8_t value[16];
    h5::read_raw(h.f, d, value);
    if (h5::get_le(value, 4) != 3)
        throw std::invalid_argument("Unsupported channel file version " + std::to_string(h5::get_le(value, 4)) +
                                    "; this version of quadriga-lib reads version 3 only.");

    // Storage index: layout, then 'Order' and 'ChannelRef' with one contiguous entry per slot
    const std::invalid_argument corrupted("Storage index in HDF file is corrupted.");
    d = root_dataset(0);
    if (d.type != h5::uint32 || d.rank != 1 || d.dims[0] != 4 || d.compact)
        throw corrupted;
    h5::read_raw(h.f, d, value);
    h.dims_addr = d.address, h.n_slot = 1;
    for (int i = 0; i < 4; ++i)
        if ((h.dims[i] = (unsigned)h5::get_le(value + 4 * i, 4)) == 0 || (h.n_slot *= h.dims[i]) > (uint64_t)UINT_MAX)
            throw corrupted;
    d = root_dataset(1);
    if (d.type != h5::uint32 || d.rank != 1 || d.dims[0] != h.n_slot || d.compact)
        throw corrupted;
    h.order_addr = d.address;
    d = root_dataset(3);
    if (d.type != h5::object_reference || d.rank != 1 || d.dims[0] != h.n_slot || d.compact)
        throw corrupted;
    h.ref_addr = d.address;
    h.index = {{h.dims_addr, 16}, {h.order_addr, 4 * h.n_slot}, {h.ref_addr, 8 * h.n_slot}};
    regions.insert(regions.end(), h.index.begin(), h.index.end());
    std::sort(regions.begin(), regions.end());
    for (size_t i = 1; i < regions.size(); ++i)
        if (qHDF_overlap(regions[i - 1], regions[i]))
            h5::malformed("the storage index overlaps the object header or data at address " + std::to_string(regions[i].first));
    return true;
}

// Linear slot number of (ix, iy, iz, iw), or -1 if the index is outside the layout
static long long qHDF_slot(const qHDF_file &h, unsigned ix, unsigned iy, unsigned iz, unsigned iw)
{
    if (ix >= h.dims[0] || iy >= h.dims[1] || iz >= h.dims[2] || iw >= h.dims[3])
        return -1LL;
    return (long long)((((uint64_t)iw * h.dims[2] + iz) * h.dims[1] + iy) * h.dims[0] + ix);
}

// Address of the channel group of a slot ('ChannelRef'); 0 if the slot is empty or outside the layout
static uint64_t qHDF_channel_ref(qHDF_file &h, long long slot)
{
    if (slot < 0)
        return 0;
    uint8_t ref[8];
    h.f.read(h.ref_addr + 8 * (uint64_t)slot, ref, 8);
    return h5::get_le(ref, 8);
}

// A group header and its decoded view
struct qHDF_group
{
    h5::object o;
    h5::group g;
};

// Count bytes read for a channel: every object header and all raw data, also when several links point to the same
// object. The sum may not exceed the end-of-file address (each header and data range of a valid file is read once),
// which bounds the memory and time of a call by the file size.
static inline void qHDF_count(qHDF_file &h, uint64_t n)
{
    if (n > h.eof || h.bytes_read > h.eof - n)
        h5::malformed("the channel references more data than the file holds");
    h.bytes_read += n;
}

static inline void qHDF_count_header(qHDF_file &h, const h5::object &o)
{
    for (const h5::chunk &c : o.chunks)
        qHDF_count(h, c.length());
}

static inline void qHDF_read_group(qHDF_file &h, uint64_t addr, qHDF_group &grp)
{
    grp.o = h5::read_object(h.f, addr, h.eof);
    qHDF_count_header(h, grp.o);
    grp.g = h5::decode_group(grp.o);
}

// Read and decode the dataset header at 'addr' (kept in 'o'); the header and contiguous data are counted (compact
// data is part of the header). An empty dataset of rank 3 becomes an arma::Cube whose slices still cost one pointer
// each: they are counted as 8 bytes per slice, and the product of the other two extents must fit 64 bits.
static h5::dataset qHDF_read_dataset(qHDF_file &h, uint64_t addr, h5::object &o)
{
    o = h5::read_object(h.f, addr, h.eof);
    qHDF_count_header(h, o);
    h5::dataset d = h5::decode_dataset(o, h.eof);
    uint64_t n = d.compact ? 0 : d.size, slice = 0;
    if (d.n_elem == 0 && d.rank == 3 && (!h5::multiply(d.dims[1], d.dims[2], slice) || !h5::multiply(d.dims[0], 8, n)))
        h5::malformed("the dataset at address " + std::to_string(addr) + " has too large extents");
    qHDF_count(h, n);
    return d;
}

// Links of a group in creation order (stable sort by the creation-order field)
static std::vector<const h5::link *> qHDF_creation_order(const h5::group &g)
{
    std::vector<const h5::link *> links;
    for (const h5::link &l : g.links)
        links.push_back(&l);
    std::stable_sort(links.begin(), links.end(), [](const h5::link *a, const h5::link *b)
                     { return a->corder < b->corder; });
    return links;
}

// The group 'Data' of the channel at a slot; false if the slot is empty or outside the layout or has no 'Data'
static bool qHDF_read_data_group(qHDF_file &h, unsigned ix, unsigned iy, unsigned iz, unsigned iw, qHDF_group &data)
{
    const uint64_t ref = qHDF_channel_ref(h, qHDF_slot(h, ix, iy, iz, iw));
    if (ref == 0)
        return false;
    qHDF_group channel;
    qHDF_read_group(h, ref, channel);
    const h5::link *l = channel.g.find("Data");
    if (l == nullptr)
        return false;
    qHDF_read_group(h, l->target, data);
    return true;
}

// Read an unstructured dataset of rank 0 (scalar) to 3 with element type T
template <typename T>
static std::any qHDF_read_numeric(qHDF_file &h, const h5::dataset &d)
{
    if (d.rank == 0)
    {
        T value{};
        h5::read_raw(h.f, d, &value);
        return value;
    }
    if (d.rank == 1)
    {
        arma::Col<T> value((arma::uword)d.dims[0], arma::fill::none);
        h5::read_raw(h.f, d, value.memptr());
        return value;
    }
    if (d.rank == 2)
    {
        arma::Mat<T> value((arma::uword)d.dims[1], (arma::uword)d.dims[0], arma::fill::none);
        h5::read_raw(h.f, d, value.memptr());
        return value;
    }
    arma::Cube<T> value((arma::uword)d.dims[2], (arma::uword)d.dims[1], (arma::uword)d.dims[0], arma::fill::none);
    h5::read_raw(h.f, d, value.memptr());
    return value;
}

// Read one unstructured dataset: a scalar dataspace gives a scalar, rank 1 an arma::Col, rank 2 an arma::Mat, rank 3
// an arma::Cube; strings are returned without trailing NUL bytes
static std::any qHDF_read_any(qHDF_file &h, uint64_t addr, std::string_view name)
{
    h5::object o;
    const h5::dataset d = qHDF_read_dataset(h, addr, o);
    switch (d.type.t)
    {
    case h5::type_t::f32:
        return qHDF_read_numeric<float>(h, d);
    case h5::type_t::f64:
        return qHDF_read_numeric<double>(h, d);
    case h5::type_t::u64:
        return qHDF_read_numeric<unsigned long long>(h, d);
    case h5::type_t::i64:
        return qHDF_read_numeric<long long>(h, d);
    case h5::type_t::u32:
        return qHDF_read_numeric<unsigned>(h, d);
    case h5::type_t::i32:
        return qHDF_read_numeric<int>(h, d);
    case h5::type_t::str:
        if (d.rank == 0)
        {
            std::string value((size_t)d.size, '\0');
            h5::read_raw(h.f, d, &value[0]);
            while (!value.empty() && value.back() == '\0')
                value.pop_back();
            return value;
        }
        break;
    default:
        break;
    }
    h5::malformed("unstructured dataset '" + std::string(name) + "' has an unsupported type or shape");
}

// Attribute values of a channel group; the defaults are those of a group without structured data
struct qHDF_channel_attributes
{
    uint32_t n_snap = 0, n_tx = 0, n_rx = 0;
    int32_t initial_position = 0;
    float center_frequency = NAN;                    // NaN = not set
    float position[4][3] = {{NAN, NAN, NAN},         // tx_position
                            {NAN, NAN, NAN},         // rx_position
                            {0.0f, 0.0f, 0.0f},      // tx_orientation
                            {0.0f, 0.0f, 0.0f}};     // rx_orientation
    char name[256] = {};                             // At most 255 bytes, NUL-terminated and NUL-padded
    long long num_path = -1;                         // 'NumPath' of a one-snapshot channel with snapshot data
};

using qHDF_links = std::vector<std::pair<std::string_view, uint64_t>>;

// Names of the position and orientation attributes and datasets, in the order of qHDF_channel_attributes::position
static const char *const qHDF_position_names[4] = {"tx_position", "rx_position", "tx_orientation", "rx_orientation"};

// Emit a group 'Data': the links to the unstructured datasets with the creation order 0, 1, ...
static uint64_t qHDF_emit_data_group(h5::output &out, h5::header &hdr, const qHDF_links &links)
{
    hdr.clear();
    hdr.link_info(true, links.size());
    hdr.group_info();
    for (size_t i = 0; i < links.size(); ++i)
        hdr.link(links[i].first, links[i].second, i);
    const uint64_t addr = hdr.emit(out.buf, out.base, h5::continuation_size);
    out.flush_if_full();
    return addr;
}

// Emit one unstructured dataset (name and type checked by qHDF_check_par): scalars get a scalar dataspace, vectors
// rank 1 (also with a single element; arma::Row too), matrices rank 2, cubes rank 3; strings are fixed-length
// strings of their bytes, an empty string is one NUL byte
static uint64_t qHDF_emit_par(h5::output &out, const std::any *par_data)
{
    unsigned long long d[3];
    void *ptr = nullptr;
    const int datatype_id = quadriga_lib::any_type_id(par_data, d, &ptr);
    if (datatype_id == 9)
    {
        const std::string *str = std::any_cast<std::string>(par_data);
        return h5::emit_dataset(out, h5::string_type((uint32_t)std::max(str->size(), (size_t)1)), 0, nullptr,
                                str->empty() ? nullptr : (const uint8_t *)str->data(), std::max(str->size(), (size_t)1));
    }
    h5::dtype t = h5::float32;
    qHDF_par_type(datatype_id, t);
    uint64_t dims[3] = {d[0], 1, 1};
    int rank = 1;
    if (datatype_id < 20)
        rank = 0;
    else if (datatype_id < 30)
        rank = 2, dims[0] = d[1], dims[1] = d[0];
    else if (datatype_id < 40)
        rank = 3, dims[0] = d[2], dims[1] = d[1], dims[2] = d[0];
    return h5::emit_dataset(out, t, rank, dims, (const uint8_t *)ptr, d[0] * d[1] * d[2] * t.size);
}

// New channel ID: one above the ID of the last root link (the largest ID ever linked)
static uint32_t qHDF_new_channel_id(const h5::root_links &root)
{
    if (root.last_id() >= (uint64_t)UINT_MAX)
        throw std::runtime_error("No free channel ID left in the HDF5 file.");
    return (uint32_t)root.last_id() + 1;
}

// Write the superblock with the new end-of-file address
static inline void qHDF_write_superblock(qHDF_file &h, uint64_t eof)
{
    const std::vector<uint8_t> sb = h5::superblock_bytes(eof, h.root.addr);
    h.f.write(0, sb.data(), sb.size());
}

static inline void qHDF_write_patch(qHDF_file &h, const h5::patch &p)
{
    h.f.write(p.addr, p.bytes.data(), p.bytes.size());
}

// Emit the header of a channel group: link info, group info, the attributes, then the links ('Data' last if
// 'data_addr' is not 0). Without 'Data', the header keeps room for the 'Data' link (hdf5_write_dset adds it later).
static uint64_t qHDF_emit_channel_group(h5::output &out, h5::header &hdr, const qHDF_channel_attributes &a,
                                        const qHDF_links &links, uint64_t data_addr)
{
    const uint64_t three = 3;
    hdr.clear();
    hdr.link_info(false);
    hdr.group_info();
    hdr.attribute("NumSnap", h5::uint32, 0, nullptr, &a.n_snap);
    hdr.attribute("NumTx", h5::uint32, 0, nullptr, &a.n_tx);
    hdr.attribute("NumRx", h5::uint32, 0, nullptr, &a.n_rx);
    hdr.attribute("Initial_position", h5::int32, 0, nullptr, &a.initial_position);
    hdr.attribute("CenterFrequency", h5::float32, 0, nullptr, &a.center_frequency);
    for (int k = 0; k < 4; ++k)
        hdr.attribute(qHDF_position_names[k], h5::float32, 1, &three, a.position[k]);
    hdr.attribute("Name", h5::string_type(256), 0, nullptr, a.name);
    if (a.num_path >= 0)
    {
        const uint32_t num_path = (uint32_t)a.num_path;
        hdr.attribute("NumPath", h5::uint32, 0, nullptr, &num_path);
    }
    for (const auto &l : links)
        hdr.link(l.first, l.second);
    if (data_addr != 0)
        hdr.link("Data", data_addr);
    const uint64_t addr = hdr.emit(out.buf, out.base, data_addr != 0 ? 0 : h5::link_message_size(4, false));
    out.flush_if_full();
    return addr;
}

// Complete the block of a new channel and update the file: the root link chunks and the channel group (with ID 'id')
// are appended to 'out'; then the block is written and the superblock, the root link, 'ChannelRef' and 'Order' of the
// slot, and the removal of the link to the old channel group 'old' (if not 0) follow in this order
static void qHDF_commit_channel(qHDF_file &h, h5::root_links &root, h5::output &out, h5::header &hdr, long long slot,
                                uint32_t id, uint64_t old, const qHDF_channel_attributes &a, const qHDF_links &links,
                                uint64_t data_addr)
{
    const std::string name = "channel_" + std::to_string(id);
    const uint64_t group_addr = out.addr() + root.insert_size(name.size());
    const h5::patch link = root.prepare_insert(name, group_addr, out.buf, out.base);
    if (qHDF_emit_channel_group(out, hdr, a, links, data_addr) != group_addr)
        throw std::logic_error("Channel group address mismatch.");
    out.flush();

    qHDF_write_superblock(h, out.addr());
    qHDF_write_patch(h, link);
    uint8_t entry[8];
    h5::put_le(entry, group_addr, 8);
    h.f.write(h.ref_addr + 8 * (uint64_t)slot, entry, 8);
    h5::put_le(entry, id, 4);
    h.f.write(h.order_addr + 4 * (uint64_t)slot, entry, 4);
    h5::patch removal;
    if (old != 0 && root.prepare_remove(old, removal))
        qHDF_write_patch(h, removal);
}

/*!MD
# hdf5_create
Create a new HDF5 channel file with a defined storage layout

- Initializes a new HDF5 file for storing wireless channel data; overwrites existing files.
- Defines a 4D layout (x, y, z, w) where each index combination maps to one channel storage slot.
- Typical dimension mapping: x = BS, y = UE, z = frequency, w = scenario/repetition.
- Layout can be reshaped later with [[hdf5_reshape_layout]] if the total slot count stays constant.
- The total slot count `nx × ny × nz × nw` must not exceed 4294967295 (2^32 - 1); larger layouts throw before
  the file is created.
- Files use the quadriga-lib channel file format version 3, an HDF5 file readable by any HDF5 reader (h5py, MATLAB,
  HDFView). Modify files only with the quadriga-lib functions: files changed by other HDF5 tools are not supported
  and may be rejected.

## Declaration:
```
void quadriga_lib::hdf5_create(
    std::string fn,
    unsigned nx = 65536,
    unsigned ny = 1,
    unsigned nz = 1,
    unsigned nw = 1);
```

## Inputs:
- **`fn`** — Path and filename of the HDF5 file to create
- **`nx`** *(optional)* — Size of x-dimension
- **`ny`** *(optional)* — Size of y-dimension
- **`nz`** *(optional)* — Size of z-dimension
- **`nw`** *(optional)* — Size of w-dimension

## See also:
- [[hdf5_reshape_layout]] (change layout dimensions of an existing file)
- [[hdf5_write]] (write channel data into a slot)
MD!*/

void quadriga_lib::hdf5_create(std::string fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    if (nx == 0 || ny == 0 || nz == 0 || nw == 0)
        throw std::invalid_argument("Storage layout dimension sizes cannot be 0.");

    // Checked before anything is allocated or written
    const uint64_t n_slot = qHDF_slot_count(nx, ny, nz, nw);

    h5::file f;
    if (!f.open(fn, "wb"))
        throw std::invalid_argument("Error creating file. It may be already opened.");

    // Root chunk 0 with the links to the root datasets and room for the directory of the root links; the datasets
    // follow it: 'ChannelDims', 'Version', 'Order' (all slots empty), 'ChannelRef' (null references)
    h5::header root;
    uint64_t addr[4] = {0, 0, 0, 0};
    auto build_root = [&]()
    {
        root.clear();
        root.link_info(false);
        root.group_info();
        root.link("ChannelDims", addr[0]);
        root.link("Order", addr[1]);
        root.link("Version", addr[2]);
        root.link("ChannelRef", addr[3]);
    };
    const size_t directory = h5::root_directory_entries * h5::continuation_size;
    build_root();

    h5::output out(f, h5::superblock_size + root.size(directory));
    const unsigned channel_dims[4] = {nx, ny, nz, nw}, version = 3;
    uint64_t extent = 4;
    addr[0] = h5::emit_dataset(out, h5::uint32, 1, &extent, (const uint8_t *)channel_dims, 16, true);
    addr[2] = h5::emit_dataset(out, h5::uint32, 0, nullptr, (const uint8_t *)&version, 4);
    extent = n_slot;
    addr[1] = h5::emit_dataset(out, h5::uint32, 1, &extent, (const uint8_t *)nullptr, 4 * n_slot, true);
    addr[3] = h5::emit_dataset(out, h5::object_reference, 1, &extent, (const uint8_t *)nullptr, 8 * n_slot, true);
    out.flush();

    build_root();
    std::vector<uint8_t> head = h5::superblock_bytes(out.addr(), h5::superblock_size);
    root.emit(head, 0, directory);
    f.write(0, head.data(), head.size());
    f.close();
}

/*!MD
# hdf5_read_layout
Read the storage layout of an HDF5 channel file

- Returns `{nx, ny, nz, nw}` describing the 4D slot grid of the file.
- Returns `{0, 0, 0, 0}` if the file does not exist; throws if the file exists but is not a channel file of
  version 3 (see [[hdf5_read_channel]]).
- `channelID` entries are `0` for empty slots; length equals `nx × ny × nz × nw` (serialized linear index).

## Declaration:
```
arma::u32_vec quadriga_lib::hdf5_read_layout(
    std::string fn,
    arma::u32_vec *channelID = nullptr);
```

## Inputs:
- **`fn`** — Path to the HDF5 file
- **`channelID`** *(optional)* — Pointer to vector receiving the serialized slot occupancy list; `[nx·ny·nz·nw]`

## Returns:
- Four-element vector `{nx, ny, nz, nw}` describing the layout dimensions

## See also:
- [[hdf5_create]] (create a file with a defined layout)
- [[hdf5_reshape_layout]] (change layout dimensions of an existing file)
MD!*/

arma::u32_vec quadriga_lib::hdf5_read_layout(std::string fn, arma::u32_vec *channelID)
{
    qHDF_file h;
    if (!qHDF_open(h, fn, false))
        return arma::u32_vec(4, arma::fill::zeros);
    if (channelID != nullptr)
    {
        if (channelID->n_elem != (arma::uword)h.n_slot)
            channelID->set_size((arma::uword)h.n_slot);
        h.f.read(h.order_addr, channelID->memptr(), (size_t)(4 * h.n_slot));
    }
    return arma::u32_vec({h.dims[0], h.dims[1], h.dims[2], h.dims[3]});
}

// Check the structured data of a channel before anything is written (the checks of hdf5_write in their order)
template <typename dtype>
static void qHDF_check_channel(const quadriga_lib::channel<dtype> *ch)
{
    const arma::uword n_snap = ch->n_snap();
    if (n_snap > (arma::uword)UINT_MAX)
        throw std::invalid_argument("Too many snapshots.");
    if (ch->n_path().n_elem != n_snap)
        throw std::invalid_argument("Number of elements returned by 'n_path()' does not match number of snapshots.");
    const std::pair<const arma::Mat<dtype> *, const char *> positions[4] = {
        {&ch->tx_pos, "tx_pos"}, {&ch->rx_pos, "rx_pos"}, {&ch->tx_orientation, "tx_orientation"}, {&ch->rx_orientation, "rx_orientation"}};
    for (const auto &p : positions)
        if (p.first->n_elem != 0 && p.first->n_rows != 3)
            throw std::invalid_argument("'" + std::string(p.second) + "' must have 3 rows.");
    if (ch->coeff_re.size() == (size_t)n_snap && ch->coeff_im.size() == (size_t)n_snap)
        for (size_t i = 0; i < (size_t)n_snap; ++i)
            if (arma::size(ch->coeff_re[i]) != arma::size(ch->coeff_im[i]))
                throw std::invalid_argument("Size mismatch between 'coeff_re[" + std::to_string(i) + "]' and 'coeff_im[" +
                                            std::to_string(i) + "]'.");
}

// Emit the structured data of a channel (checked by qHDF_check_channel) and set the attribute values
// - Snapshot data: a field is written for every snapshot if it has one entry per snapshot, also for snapshots without
//   paths (datasets with extent 0). Several snapshots: the datasets of snapshot i, then its group 'Snap_<i>' with the
//   attribute 'NumPath'. One snapshot: the datasets go into the channel group, 'NumPath' becomes a channel attribute.
// - Positions, orientations and center frequency: one value is an attribute, one value per snapshot a dataset (the
//   attribute keeps its default then); empty or other sizes keep the default
// - Single precision: double input is converted
// - The links of the channel group are appended to 'links' (names of snapshot groups are kept in 'snap_names')
template <typename dtype>
static void qHDF_emit_channel_data(h5::output &out, h5::header &hdr, const quadriga_lib::channel<dtype> *ch,
                                   qHDF_channel_attributes &a, qHDF_links &links, std::vector<std::string> &snap_names)
{
    const size_t n_snap = (size_t)ch->n_snap();
    const arma::uvec n_path = ch->n_path();
    a.n_snap = (uint32_t)n_snap, a.n_tx = (uint32_t)ch->n_tx(), a.n_rx = (uint32_t)ch->n_rx();
    a.initial_position = ch->initial_position;
    std::memcpy(a.name, ch->name.data(), std::min(ch->name.size(), (size_t)255));
    if (ch->center_frequency.n_elem == 1)
        a.center_frequency = (float)ch->center_frequency.at(0);
    const arma::Mat<dtype> *positions[4] = {&ch->tx_pos, &ch->rx_pos, &ch->tx_orientation, &ch->rx_orientation};
    for (int k = 0; k < 4; ++k)
        if (positions[k]->n_rows == 3 && positions[k]->n_cols == 1)
            for (int i = 0; i < 3; ++i)
                a.position[k][i] = (float)positions[k]->at(i);

    // Single-precision dataset of rank 1 to 3; extents slowest first
    auto f32 = [&](int rank, uint64_t d0, uint64_t d1, uint64_t d2, const dtype *data)
    {
        const uint64_t dims[3] = {d0, d1, d2};
        uint64_t n_elem = 1;
        for (int i = 0; i < rank; ++i)
            n_elem *= dims[i];
        return h5::emit_dataset(out, h5::float32, rank, dims, data, 4 * n_elem);
    };

    const bool write_coeff = ch->coeff_re.size() == n_snap && ch->coeff_im.size() == n_snap,
               write_delay = ch->delay.size() == n_snap,
               write_path_gain = ch->path_gain.size() == n_snap,
               write_path_length = ch->path_length.size() == n_snap,
               write_path_polarization = ch->path_polarization.size() == n_snap,
               write_path_angles = ch->path_angles.size() == n_snap,
               write_path_fbs_pos = ch->path_fbs_pos.size() == n_snap,
               write_path_lbs_pos = ch->path_lbs_pos.size() == n_snap,
               write_no_interact = ch->no_interact.size() == n_snap,
               write_interact_coord = ch->interact_coord.size() == n_snap;
    const bool snapshot_data = n_snap != 0 &&
                               (write_coeff || write_delay || write_path_gain || write_path_length || write_path_polarization ||
                                write_path_angles || write_path_fbs_pos || write_path_lbs_pos || write_no_interact || write_interact_coord);

    // The datasets of snapshot i, in link order
    auto emit_snapshot = [&](size_t i, qHDF_links &dst)
    {
        auto cube = [&](const char *name, const arma::Cube<dtype> &x)
        { dst.push_back({name, f32(3, x.n_slices, x.n_cols, x.n_rows, x.memptr())}); };
        auto mat = [&](const char *name, const arma::Mat<dtype> &x)
        { dst.push_back({name, f32(2, x.n_cols, x.n_rows, 1, x.memptr())}); };
        auto col = [&](const char *name, const arma::Col<dtype> &x)
        { dst.push_back({name, f32(1, x.n_elem, 1, 1, x.memptr())}); };
        if (write_coeff)
            cube("coeff_re", ch->coeff_re[i]), cube("coeff_im", ch->coeff_im[i]);
        if (write_delay)
            cube("delay", ch->delay[i]);
        if (write_path_gain)
            col("path_gain", ch->path_gain[i]);
        if (write_path_length)
            col("path_length", ch->path_length[i]);
        if (write_path_polarization)
            mat("path_polarization", ch->path_polarization[i]);
        if (write_path_angles)
            mat("path_angles", ch->path_angles[i]);
        if (write_path_fbs_pos)
            mat("path_fbs_pos", ch->path_fbs_pos[i]);
        if (write_path_lbs_pos)
            mat("path_lbs_pos", ch->path_lbs_pos[i]);
        if (write_no_interact)
        {
            const uint64_t n = ch->no_interact[i].n_elem;
            dst.push_back({"no_interact", h5::emit_dataset(out, h5::uint32, 1, &n, ch->no_interact[i].memptr(), 4 * n)});
        }
        if (write_interact_coord)
            mat("interact_coord", ch->interact_coord[i]);
    };

    // Several snapshots: datasets and group of each snapshot
    std::vector<uint64_t> snap_addr;
    if (snapshot_data && n_snap > 1)
    {
        qHDF_links snap_links;
        snap_names.reserve(n_snap); // The links keep views of the names
        snap_addr.reserve(n_snap);
        for (size_t i = 0; i < n_snap; ++i)
        {
            snap_links.clear();
            emit_snapshot(i, snap_links);
            const uint32_t num_path = (uint32_t)n_path.at(i);
            hdr.clear();
            hdr.link_info(false);
            hdr.group_info();
            hdr.attribute("NumPath", h5::uint32, 0, nullptr, &num_path);
            for (const auto &l : snap_links)
                hdr.link(l.first, l.second);
            snap_addr.push_back(hdr.emit(out.buf, out.base, 0));
            out.flush_if_full();
            snap_names.push_back("Snap_" + std::to_string(i));
        }
    }

    // Values per snapshot
    if (n_snap > 1 && ch->center_frequency.n_elem == n_snap)
        links.push_back({"CenterFrequency", f32(1, n_snap, 1, 1, ch->center_frequency.memptr())});
    for (int k = 0; k < 4; ++k)
        if (n_snap > 1 && positions[k]->n_rows == 3 && positions[k]->n_cols == n_snap)
            links.push_back({qHDF_position_names[k], f32(2, n_snap, 3, 1, positions[k]->memptr())});

    // One snapshot: its datasets in the channel group
    if (snapshot_data && n_snap == 1)
    {
        emit_snapshot(0, links);
        a.num_path = (long long)n_path.at(0);
    }
    for (size_t i = 0; i < snap_addr.size(); ++i)
        links.push_back({snap_names[i], snap_addr[i]});
}

/*!MD
# hdf5_write
Write a channel object to an HDF5 file at a specified 4D index

- Writes a `quadriga_lib::channel<dtype>` object to slot `(ix, iy, iz, iw)` in the HDF5 file.
- Creates the file with default layout `(65536 × 1 × 1 × 1)` if it does not exist; an existing file must be a
  channel file of version 3 written by quadriga-lib (files of quadriga-lib 0.12.4 and earlier are rejected, see
  [[hdf5_read_channel]]).
- If the slot already holds data, the slot is replaced completely: the old structured data and all unstructured
  datasets of the slot (`par_*`, including those added with [[hdf5_write_dset]]) are removed, and only the new
  channel is stored there. The space of the removed data is not reused, so the file does not shrink; to reclaim
  it, copy the channels into a new file ([[hdf5_read_channel]], [[hdf5_write]]).
- Throws if the index was not reserved during [[hdf5_create]].
- Structured data is always stored in single precision; input may be float or double.
- Unstructured data: supported types are string, double, float, (u)int32, (u)int64; up to 3D; storage order preserved.
- Set `assume_valid = true` to skip integrity validation (faster but unsafe for potentially corrupted data).
- The channel name is stored with at most 255 characters.
- One writer per file at a time; files are not locked. Reading is safe while no one writes; different files can be
  read and written from parallel threads.

## Declaration:
```
int quadriga_lib::hdf5_write(
    const quadriga_lib::channel<dtype> *ch,
    std::string fn,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    bool assume_valid = false);
```

## Inputs:
- **`ch`** — Pointer to the channel object to write
- **`fn`** — Path to the HDF5 file
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`assume_valid`** *(optional)* — Skip channel integrity validation before writing

## Returns:
- `0` if the slot was empty, `1` if existing data in the slot was replaced

## See also:
- [[hdf5_create]] (create file and reserve layout)
- [[hdf5_read_channel]] (read channel data from a slot)
MD!*/

template <typename dtype>
int quadriga_lib::hdf5_write(const quadriga_lib::channel<dtype> *ch, std::string fn,
                             unsigned ix, unsigned iy, unsigned iz, unsigned iw, bool assume_valid)
{
    // Validate the channel data
    bool channel_is_empty = false;
    if (!assume_valid)
    {
        std::string error_message = ch->is_valid();
        if (error_message.length() != 0)
            throw std::invalid_argument(error_message.c_str());

        // Determine if the channel object has no structured data fields
        channel_is_empty = ch->empty();
    }
    if (ch->par_names.size() != ch->par_data.size())
        throw std::invalid_argument("Number of elements in 'par_data' must match number of elements in 'par_name'.");

    // Are there unstructured fields?
    if (channel_is_empty && ch->par_names.size() == 0)
        return 0; // Nothing to be done!

    // Names and types of the unstructured data are checked before the file is changed
    std::vector<std::string> names; // Dataset names with the prefix
    std::set<std::string> unique;
    for (size_t i = 0; i < ch->par_names.size(); ++i)
    {
        names.push_back("par_" + ch->par_names[i]);
        qHDF_check_par(names[i], &ch->par_data[i]);
        if (!unique.insert(names[i]).second)
            throw std::invalid_argument("Dataset '" + names[i] + "' already exists.");
    }

    // Open the file (created with the default layout if it does not exist), the slot and its old channel group
    const char *open_error = "Error opening file. It may be opened by another program.";
    qHDF_file h;
    if (!qHDF_open(h, fn, true, open_error))
    {
        quadriga_lib::hdf5_create(fn);
        if (!qHDF_open(h, fn, true, open_error)) // Only if the file was removed in between
            throw std::invalid_argument(open_error);
    }
    const long long slot = qHDF_slot(h, ix, iy, iz, iw);
    if (slot < 0)
        throw std::invalid_argument("Index out of bound.");
    const uint64_t old = qHDF_channel_ref(h, slot);
    h5::root_links root("channel_");
    root.load(h.f, h.root, h.eof, old);
    qHDF_check_chunks(h, root.ranges());
    if (!channel_is_empty)
        qHDF_check_channel(ch);
    const uint32_t id = qHDF_new_channel_id(root);

    // The new channel is written as one block at the end of the file: structured data, unstructured data and their
    // group 'Data', the root link chunks, and the channel group
    h5::output out(h.f, h.eof);
    h5::header hdr;
    qHDF_channel_attributes a;
    qHDF_links links;
    std::vector<std::string> snap_names;
    if (!channel_is_empty)
        qHDF_emit_channel_data(out, hdr, ch, a, links, snap_names);
    uint64_t data_addr = 0;
    if (!names.empty())
    {
        qHDF_links par_links;
        for (size_t i = 0; i < names.size(); ++i)
            par_links.push_back({names[i], qHDF_emit_par(out, &ch->par_data[i])});
        data_addr = qHDF_emit_data_group(out, hdr, par_links);
    }
    qHDF_commit_channel(h, root, out, hdr, slot, id, old, a, links, data_addr);
    h.f.close();
    return old != 0 ? 1 : 0;
}

template int quadriga_lib::hdf5_write(const quadriga_lib::channel<float> *ch, std::string fn,
                                      unsigned ix, unsigned iy, unsigned iz, unsigned iw, bool assume_valid);

template int quadriga_lib::hdf5_write(const quadriga_lib::channel<double> *ch, std::string fn,
                                      unsigned ix, unsigned iy, unsigned iz, unsigned iw, bool assume_valid);

/*!MD
# hdf5_read_channel
Read a channel object from an HDF5 file at a specified 4D index

- Returns an empty channel object (`n_snap() == 0`) if the slot is empty or outside the layout.
- Structured data is stored in single precision in the file and converted to `dtype` on read.
- Unstructured fields (`std::any`) retain their original stored type without conversion; see [[hdf5_read_dset]]
  for the returned types.
- Round trip with [[hdf5_write]]: every per-snapshot field comes back with one entry per snapshot, also for
  snapshots without paths (empty entries keep their shape); a channel written without `center_frequency` comes back
  with an empty `center_frequency`; empty `tx_orientation` / `rx_orientation` come back as `[3, 1]` zeros.
- Throws `std::invalid_argument` if the file is malformed (e.g. a dataset does not have the size given by the
  attributes of the channel, or a required dataset or attribute is missing) or is not a channel file of version 3
  written by quadriga-lib: files written by quadriga-lib 0.12.4 and earlier (version 2) are rejected; files modified
  with other HDF5 tools are not supported and may be rejected.

## Declaration:
```
quadriga_lib::channel<dtype> quadriga_lib::hdf5_read_channel(
    std::string fn,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0);
```

## Inputs:
- **`fn`** — Path to the HDF5 file
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension

## Returns:
- Channel object at the specified slot; empty if no data is present

## See also:
- [[hdf5_write]] (write a channel object to a slot)
- [[hdf5_read_layout]] (inspect slot occupancy before reading)
MD!*/

template <typename dtype>
quadriga_lib::channel<dtype> quadriga_lib::hdf5_read_channel(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw)
{
    qHDF_file h;
    if (!qHDF_open(h, fn, false))
        throw std::invalid_argument("File does not exist.");
    const uint64_t ref = qHDF_channel_ref(h, qHDF_slot(h, ix, iy, iz, iw));
    if (ref == 0)
        return quadriga_lib::channel<dtype>();

    qHDF_group channel;
    qHDF_read_group(h, ref, channel);
    const h5::group &g = channel.g;
    quadriga_lib::channel<dtype> c;

    // Unstructured data: the datasets 'par_*' of the group 'Data' in creation order
    if (const h5::link *data = g.find("Data"))
    {
        qHDF_group data_group;
        qHDF_read_group(h, data->target, data_group);
        for (const h5::link *l : qHDF_creation_order(data_group.g))
            if (l->name.compare(0, 4, "par_") == 0)
            {
                c.par_data.push_back(qHDF_read_any(h, l->target, l->name));
                c.par_names.push_back(std::string(l->name.substr(4)));
            }
    }

    // Attribute of a channel or snapshot group with exactly the type 't' and 'n' values; nullptr if it is missing and
    // not required
    auto attribute = [](const h5::group &grp, const char *name, h5::dtype t, uint64_t n, bool required) -> const uint8_t *
    {
        const h5::attribute *a = grp.find_attribute(name);
        auto fail = [&](const std::string &what)
        { h5::malformed("attribute '" + std::string(name) + "' " + what); };
        if (a == nullptr && required)
            fail("is missing");
        if (a == nullptr)
            return nullptr;
        if (a->type != t)
            fail("has an unexpected type");
        if (a->n_elem != n)
            fail("must have " + std::to_string(n) + (n == 1 ? " value" : " values"));
        return a->value;
    };
    auto read_u32 = [&](const h5::group &grp, const char *name, bool required) -> uint32_t
    {
        const uint8_t *v = attribute(grp, name, h5::uint32, 1, required);
        return v == nullptr ? 0 : (uint32_t)h5::get_le(v, 4);
    };

    // Channel name (the bytes before the first NUL), number of snapshots, index of the reference position
    if (const h5::attribute *a = g.find_attribute("Name"))
    {
        if (a->type != h5::string_type(256) || a->n_elem != 1)
            h5::malformed("attribute 'Name' must be a fixed-length string of 256 bytes");
        c.name.assign((const char *)a->value, (size_t)(std::find(a->value, a->value + 256, 0) - a->value));
    }
    const uint32_t n_snapshots = read_u32(g, "NumSnap", true);
    if (const uint8_t *v = attribute(g, "Initial_position", h5::int32, 1, false))
        c.initial_position = (int)(int32_t)h5::get_le(v, 4);
    if (n_snapshots == 0)
        return c;

    // Number of transmitters and receivers: sizes of the coefficient datasets
    const uint32_t n_tx = read_u32(g, "NumTx", true), n_rx = read_u32(g, "NumRx", true);
    const uint64_t n_snap = n_snapshots;

    // Dataset of a group with the element type 't' and the extents 'expected' (slowest first; h5::undef: not checked);
    // false if the group has no such link
    h5::object o;
    h5::dataset d;
    std::vector<float> buffer;
    auto find_dataset = [&](const h5::group &grp, const char *name, h5::dtype t, std::initializer_list<uint64_t> expected)
    {
        const h5::link *l = grp.find(name);
        if (l == nullptr)
            return false;
        d = qHDF_read_dataset(h, l->target, o);
        auto fail = [&](const std::string &what)
        { h5::malformed("dataset '" + std::string(name) + "' " + what); };
        if (d.type != t)
            fail("has an unexpected type");
        if (d.rank != (int)expected.size())
            fail("has " + std::to_string(d.rank) + " dimensions, expected " + std::to_string(expected.size()));
        int i = 0;
        for (uint64_t e : expected)
        {
            if (e != h5::undef && d.dims[i] != e)
                fail("has extent " + std::to_string(d.dims[i]) + " in dimension " + std::to_string(i) + ", expected " +
                     std::to_string(e));
            ++i;
        }
        return true;
    };
    // Read the single-precision data of 'd' as dtype
    auto read_float = [&](dtype *dst)
    {
        if constexpr (std::is_same<dtype, float>::value)
            h5::read_raw(h.f, d, dst);
        else
        {
            buffer.resize((size_t)d.n_elem);
            h5::read_raw(h.f, d, buffer.data());
            for (size_t i = 0; i < buffer.size(); ++i)
                dst[i] = (dtype)buffer[i];
        }
    };

    // Center frequency in [Hz]: dataset of length [n_snap] or scalar attribute (NaN = not set, read back empty)
    if (find_dataset(g, "CenterFrequency", h5::float32, {n_snap}))
    {
        c.center_frequency.set_size((arma::uword)n_snap);
        read_float(c.center_frequency.memptr());
    }
    else if (const uint8_t *v = attribute(g, "CenterFrequency", h5::float32, 1, false))
    {
        float value;
        std::memcpy(&value, v, 4);
        if (!std::isnan(value))
            c.center_frequency = {(dtype)value};
    }

    // Positions and orientations: dataset [3, n_snap] or attribute [3, 1]; orientations are optional
    auto read_position = [&](const char *name, arma::Mat<dtype> &value, bool required)
    {
        if (find_dataset(g, name, h5::float32, {n_snap, 3}))
        {
            value.set_size(3, (arma::uword)n_snap);
            read_float(value.memptr());
        }
        else if (const uint8_t *v = attribute(g, name, h5::float32, 3, required))
        {
            float data[3];
            std::memcpy(data, v, 12);
            value.set_size(3, 1);
            value.at(0) = (dtype)data[0], value.at(1) = (dtype)data[1], value.at(2) = (dtype)data[2];
        }
    };
    read_position("tx_position", c.tx_pos, true);
    read_position("rx_position", c.rx_pos, true);
    read_position("tx_orientation", c.tx_orientation, false);
    read_position("rx_orientation", c.rx_orientation, false);

    // Snapshot data: in the channel group for one snapshot; otherwise in the groups 'Snap_0' ... 'Snap_<n_snap - 1>'
    // (all of them or none). The links are collected (their number is bounded by the file) and sorted by index: with
    // n_snap links of indices below n_snap, the sorted indices are 0, 1, ... unless one is linked twice.
    std::vector<std::pair<uint64_t, uint64_t>> snaps; // Snapshot index and group address
    if (n_snap > 1)
    {
        for (const h5::link &l : g.links)
            if (l.name.compare(0, 5, "Snap_") == 0)
            {
                const long long i = h5::parse_decimal(l.name.substr(5));
                if (i < 0 || (uint64_t)i >= n_snap)
                    h5::malformed("invalid snapshot group '" + std::string(l.name) + "'");
                snaps.push_back({(uint64_t)i, l.target});
            }
        if (!snaps.empty() && snaps.size() != n_snap)
            h5::malformed("the channel has " + std::to_string(snaps.size()) + " snapshot groups, expected " + std::to_string(n_snap));
        std::sort(snaps.begin(), snaps.end());
        for (size_t i = 0; i < snaps.size(); ++i)
            if (snaps[i].first != i)
                h5::malformed("snapshot group 'Snap_" + std::to_string(snaps[i].first) + "' is linked twice");
    }

    const char *const fields[11] = {"coeff_re", "coeff_im", "delay", "path_gain", "path_length", "path_polarization",
                                    "path_angles", "path_fbs_pos", "path_lbs_pos", "no_interact", "interact_coord"};
    const size_t n_read = n_snap == 1 ? 1 : snaps.size();
    unsigned present_in_first = 0;
    qHDF_group snap;
    for (size_t s = 0; s < n_read; ++s)
    {
        const h5::group *sg = &g;
        if (n_snap > 1)
        {
            qHDF_read_group(h, snaps[s].second, snap);
            sg = &snap.g;
        }

        // Every snapshot has the datasets of the first one
        unsigned present = 0;
        for (unsigned k = 0; k < 11; ++k)
            if (sg->find(fields[k]) != nullptr)
                present |= 1u << k;
        if (s == 0)
            present_in_first = present;
        else if (present != present_in_first)
            h5::malformed("snapshot " + std::to_string(s) + " has other datasets than snapshot 0");
        if (((present & 1u) != 0) != ((present & 2u) != 0))
            h5::malformed("datasets 'coeff_re' and 'coeff_im' must both be present");

        // Number of paths (missing if the snapshot holds no path data)
        const uint64_t np = read_u32(*sg, "NumPath", false);
        const arma::uword n_path = (arma::uword)np;

        // Channel coefficients, tensors of size [n_rx, n_tx, n_path]
        if (find_dataset(*sg, "coeff_re", h5::float32, {np, n_tx, n_rx}))
        {
            c.coeff_re.emplace_back((arma::uword)n_rx, (arma::uword)n_tx, n_path, arma::fill::none);
            read_float(c.coeff_re.back().memptr());
            find_dataset(*sg, "coeff_im", h5::float32, {np, n_tx, n_rx});
            c.coeff_im.emplace_back((arma::uword)n_rx, (arma::uword)n_tx, n_path, arma::fill::none);
            read_float(c.coeff_im.back().memptr());
        }

        // Path delays in seconds, tensors of size [n_rx, n_tx, n_path] or [1, 1, n_path]
        if (find_dataset(*sg, "delay", h5::float32, {np, h5::undef, h5::undef}))
        {
            c.delay.emplace_back((arma::uword)d.dims[2], (arma::uword)d.dims[1], n_path, arma::fill::none);
            read_float(c.delay.back().memptr());
        }

        // Path gain and absolute path length, vectors of length [n_path]
        if (find_dataset(*sg, "path_gain", h5::float32, {np}))
        {
            c.path_gain.emplace_back(n_path, arma::fill::none);
            read_float(c.path_gain.back().memptr());
        }
        if (find_dataset(*sg, "path_length", h5::float32, {np}))
        {
            c.path_length.emplace_back(n_path, arma::fill::none);
            read_float(c.path_length.back().memptr());
        }

        // Polarization transfer function [8, n_path], angles [n_path, 4], scatterer positions [3, n_path]
        if (find_dataset(*sg, "path_polarization", h5::float32, {np, 8}))
        {
            c.path_polarization.emplace_back(8, n_path, arma::fill::none);
            read_float(c.path_polarization.back().memptr());
        }
        if (find_dataset(*sg, "path_angles", h5::float32, {4, np}))
        {
            c.path_angles.emplace_back(n_path, 4, arma::fill::none);
            read_float(c.path_angles.back().memptr());
        }
        if (find_dataset(*sg, "path_fbs_pos", h5::float32, {np, 3}))
        {
            c.path_fbs_pos.emplace_back(3, n_path, arma::fill::none);
            read_float(c.path_fbs_pos.back().memptr());
        }
        if (find_dataset(*sg, "path_lbs_pos", h5::float32, {np, 3}))
        {
            c.path_lbs_pos.emplace_back(3, n_path, arma::fill::none);
            read_float(c.path_lbs_pos.back().memptr());
        }

        // Number of interaction points of a path with the environment, 0 = LOS, and their coordinates [3, sum]
        if (find_dataset(*sg, "no_interact", h5::uint32, {np}))
        {
            arma::u32_vec no_interact(n_path, arma::fill::none);
            h5::read_raw(h.f, d, no_interact.memptr());
            uint64_t n_coord = 0;
            for (unsigned v : no_interact)
                n_coord += v;
            if (n_coord != 0 && !find_dataset(*sg, "interact_coord", h5::float32, {n_coord, 3})) // Checked first
                h5::malformed("dataset 'interact_coord' is missing");
            arma::Mat<dtype> coord(3, (arma::uword)n_coord, arma::fill::none);
            if (n_coord != 0)
                read_float(coord.memptr());
            c.no_interact.push_back(std::move(no_interact));
            c.interact_coord.push_back(std::move(coord));
        }
    }
    return c;
}

template quadriga_lib::channel<float> quadriga_lib::hdf5_read_channel(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw);

template quadriga_lib::channel<double> quadriga_lib::hdf5_read_channel(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw);

/*!MD
# hdf5_reshape_layout
Reshape the 4D storage layout of an existing HDF5 channel file

- Updates `{nx, ny, nz, nw}` of an existing file; total slot count `nx × ny × nz × nw` must remain unchanged.
- Throws if the new layout violates the total-count constraint or has more than 4294967295 (2^32 - 1) slots.
- Throws if the file does not exist or is not a channel file of version 3 (see [[hdf5_read_channel]]).
- One writer per file at a time; files are not locked.

## Declaration:
```
void quadriga_lib::hdf5_reshape_layout(
    std::string fn,
    unsigned nx,
    unsigned ny = 1,
    unsigned nz = 1,
    unsigned nw = 1);
```

## Inputs:
- **`fn`** — Path to the HDF5 file
- **`nx`** — New size of x-dimension
- **`ny`** *(optional)* — New size of y-dimension
- **`nz`** *(optional)* — New size of z-dimension
- **`nw`** *(optional)* — New size of w-dimension

## See also:
- [[hdf5_create]] (set initial layout at file creation)
- [[hdf5_read_layout]] (query current layout)
MD!*/

void quadriga_lib::hdf5_reshape_layout(std::string fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    if (!h5::file::exists(fn))
        throw std::invalid_argument("File does not exist.");

    // Checked in 64 bits, before the file is opened
    const unsigned long long n_data_new = qHDF_slot_count(nx, ny, nz, nw);

    qHDF_file h;
    if (!qHDF_open(h, fn, true)) // Only if the file was removed in between
        throw std::invalid_argument("File does not exist.");
    if (n_data_new != h.n_slot)
        throw std::invalid_argument("Mismatch in number of elements in storage index.");

    uint8_t channel_dims[16];
    const unsigned dims[4] = {nx, ny, nz, nw};
    for (int i = 0; i < 4; ++i)
        h5::put_le(channel_dims + 4 * i, dims[i], 4);
    h.f.write(h.dims_addr, channel_dims, 16);
    h.f.close();
}

/*!MD
# hdf5_read_dset
Read an unstructured dataset from an HDF5 file at a specified 4D index

- Reads a user-defined dataset stored under `prefix + par_name` (e.g., `"par_carrier_frequency"`).
- Returns an empty `std::any` if the dataset does not exist at the specified slot or name.
- Use [[any_type_id]] to determine the contained type and obtain a raw pointer.
- Returned types: a scalar dataset gives a scalar; a 1D dataset an `arma::Col`, also when it has a single element;
  a 2D dataset an `arma::Mat`; a 3D dataset an `arma::Cube`. An `arma::Row` written with [[hdf5_write_dset]]
  therefore comes back as an `arma::Col`.
- Element types: float, double, 32- and 64-bit signed and unsigned integers.
- Strings are returned without trailing NUL characters (an empty string written by [[hdf5_write_dset]] comes back
  empty).
- Throws if the file does not exist or is not a channel file of version 3 (see [[hdf5_read_channel]]).

## Declaration:
```
std::any quadriga_lib::hdf5_read_dset(
    std::string fn,
    std::string par_name,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    std::string prefix = "par_");
```

## Inputs:
- **`fn`** — Path to the HDF5 file
- **`par_name`** — Dataset name without prefix, e.g., `"carrier_frequency"`
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`prefix`** *(optional)* — Dataset name prefix prepended before `par_name`

## Returns:
- `std::any` containing the dataset, or empty `std::any` if not found

## See also:
- [[hdf5_write_dset]] (write an unstructured dataset)
- [[any_type_id]] (inspect the type held in a `std::any`)
MD!*/

// Read unstructured data from HDF5 file
std::any quadriga_lib::hdf5_read_dset(std::string fn, std::string par_name,
                                      unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                                      std::string prefix)
{
    qHDF_file h;
    if (!qHDF_open(h, fn, false))
        throw std::invalid_argument("File does not exist.");
    qHDF_group data;
    if (!qHDF_read_data_group(h, ix, iy, iz, iw, data))
        return std::any();
    const std::string name = prefix + par_name;
    const h5::link *l = data.g.find(name);
    return l == nullptr ? std::any() : qHDF_read_any(h, l->target, name);
}

/*!MD
# hdf5_read_dset_names
Read names of unstructured datasets stored at a 4D slot in an HDF5 file

- Finds all datasets whose HDF5 name starts with `prefix` at slot `(ix, iy, iz, iw)`; returned names exclude the prefix.
- Throws if the file does not exist or is not a channel file of version 3 (see [[hdf5_read_channel]]).

## Declaration:
```
arma::uword quadriga_lib::hdf5_read_dset_names(
    std::string fn,
    std::vector<std::string> *par_names,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    std::string prefix = "par_");
```

## Inputs:
- **`fn`** — Path to the HDF5 file
- **`par_names`** — Pointer to vector receiving dataset names (without prefix)
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`prefix`** *(optional)* — Prefix used to identify unstructured datasets

## Returns:
- Number of datasets found at the specified slot

## See also:
- [[hdf5_read_dset]] (read a dataset by name)
- [[hdf5_write_dset]] (write an unstructured dataset)
MD!*/

// Read names of the unstructured data fields from the HDF file
arma::uword quadriga_lib::hdf5_read_dset_names(std::string fn, std::vector<std::string> *par_names,
                                               unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                                               std::string prefix)
{
    qHDF_file h;
    if (!qHDF_open(h, fn, false))
        throw std::invalid_argument("File does not exist.");
    qHDF_group data;
    if (!qHDF_read_data_group(h, ix, iy, iz, iw, data))
        return 0ULL;

    arma::uword n_found = 0;
    for (const h5::link *l : qHDF_creation_order(data.g))
        if (l->name.compare(0, prefix.length(), prefix) == 0)
        {
            if (par_names != nullptr)
                par_names->push_back(std::string(l->name.substr(prefix.length())));
            ++n_found;
        }
    return n_found;
}

/*!MD
# hdf5_write_dset
Write a single unstructured dataset to an HDF5 file at a specified 4D index

- Dataset is stored under `prefix + par_name`; name must contain only alphanumeric characters and underscores.
- Adds the dataset to the slot and keeps everything else stored there (channel data and other datasets); an empty
  slot is reserved for it. A later [[hdf5_write]] to the same slot replaces the slot completely, including the
  datasets written with this function.
- Supported scalar types: `std::string`, `unsigned`, `int`, `long long`, `unsigned long long`, `float`, `double`
- Supported Armadillo types: `arma::Col`, `arma::Row`, `arma::Mat`, `arma::Cube` with element types `float`, `double`, `int`, `unsigned`, `sword`, `uword`, `unsigned long long`
- `arma::Row` is converted to `arma::Col` before writing.
- Throws for unsupported types and invalid names, before the file is changed.
- Throws if a dataset with the same name already exists at the specified slot; no overwrite/update is supported.
- One writer per file at a time; files are not locked.
- Creates the file with the default layout of [[hdf5_create]] if it does not exist; an existing file must be
  a channel file of version 3 (see [[hdf5_read_channel]]).

## Declaration:
```
void quadriga_lib::hdf5_write_dset(
    std::string fn,
    std::string par_name,
    const std::any *par_data,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    std::string prefix = "par_");
```

## Inputs:
- **`fn`** — Path to the HDF5 file
- **`par_name`** — Dataset name without prefix; alphanumeric and underscores only
- **`par_data`** — Pointer to the data to write; type must be supported (see above)
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`prefix`** *(optional)* — Prefix prepended to `par_name` in the HDF5 file

## See also:
- [[hdf5_read_dset]] (read an unstructured dataset by name)
- [[hdf5_read_dset_names]] (list available dataset names at a slot)
- [[any_type_id]] (inspect the type held in a `std::any`)
MD!*/

// Writes unstructured data to a hdf5 file
void quadriga_lib::hdf5_write_dset(std::string fn, std::string par_name, const std::any *par_data,
                                   unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                                   std::string prefix)
{
    // Name and type are checked before the file is changed
    const std::string name = prefix + par_name;
    qHDF_check_par(name, par_data);

    // Open the file (created with the default layout if it does not exist) and the slot
    qHDF_file h;
    if (!qHDF_open(h, fn, true))
    {
        quadriga_lib::hdf5_create(fn);
        if (!qHDF_open(h, fn, true)) // Only if the file was removed in between
            throw std::invalid_argument("Error opening file.");
    }
    const long long slot = qHDF_slot(h, ix, iy, iz, iw);
    if (slot < 0)
        throw std::invalid_argument("Index out of bound.");
    const uint64_t ref = qHDF_channel_ref(h, slot);
    h5::header hdr;

    // Empty slot: a new channel group with the default attributes and a group 'Data' holding the dataset
    if (ref == 0)
    {
        h5::root_links root("channel_");
        root.load(h.f, h.root, h.eof, 0);
        qHDF_check_chunks(h, root.ranges());
        const uint32_t id = qHDF_new_channel_id(root);
        h5::output out(h.f, h.eof);
        const uint64_t data_addr = qHDF_emit_data_group(out, hdr, {{name, qHDF_emit_par(out, par_data)}});
        qHDF_commit_channel(h, root, out, hdr, slot, id, 0, qHDF_channel_attributes(), {}, data_addr);
        h.f.close();
        return;
    }

    // Filled slot: the dataset is linked into the existing group 'Data', or into a new one that is linked into the
    // channel group; 'Order' and 'ChannelRef' are unchanged
    h5::chain_links channel, data;
    channel.load(h.f, ref, h.eof);
    qHDF_check_chunks(h, channel.ranges());
    const h5::link *data_link = channel.find("Data");
    if (data_link != nullptr)
    {
        data.load(h.f, data_link->target, h.eof);
        qHDF_check_chunks(h, data.ranges());
        if (data.find(name) != nullptr)
            throw std::invalid_argument("Dataset '" + name + "' already exists.");
        data.check_insert(name.size(), true);
    }
    else
        channel.check_insert(4, false);

    h5::output out(h.f, h.eof);
    const uint64_t dset_addr = qHDF_emit_par(out, par_data);
    std::vector<h5::patch> patches;
    if (data_link != nullptr)
        patches = data.prepare_insert(name, dset_addr, true, out.buf, out.base);
    else
        patches = channel.prepare_insert("Data", qHDF_emit_data_group(out, hdr, {{name, dset_addr}}), false, out.buf, out.base);
    out.flush();
    qHDF_write_superblock(h, out.addr());
    for (const h5::patch &p : patches)
        qHDF_write_patch(h, p);
    h.f.close();
}
