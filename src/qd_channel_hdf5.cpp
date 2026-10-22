// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include <hdf5.h>
#include <algorithm>
#include <climits>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# get_HDF5_version
Return the HDF5 version string as defined by the compile-time header macros

## Declaration:
```
std::string quadriga_lib::get_HDF5_version();
```

## Returns:
- Version string in the format `"x.y.z"`, e.g., `"1.12.2"`
MD!*/

// Return HDF5 version info
#define AUX(x) #x
#define STRINGIFY(x) AUX(x)
std::string quadriga_lib::get_HDF5_version()
{
    std::string str = STRINGIFY(H5_VERS_MAJOR);
    str += ".";
    str += STRINGIFY(H5_VERS_MINOR);
    str += ".";
    str += STRINGIFY(H5_VERS_RELEASE);
    return str;
}

// Internal helpers
// - Every HDF5 identifier is owned by a qHDF_id and closed when it goes out of scope, also on error paths.
// - Every HDF5 return code is checked. Malformed files give std::invalid_argument, failing writes std::runtime_error.
// - HDF5's automatic error printing is switched off while a library function runs and the caller's handler is
//   restored afterwards (qHDF_error_guard). The HDF5 library is never closed (no H5close), so the identifiers of
//   a calling program that uses HDF5 itself stay valid.

// Helper function: File exists?
static inline bool qHDF_file_exists(const std::string &name)
{
    if (FILE *file = std::fopen(name.c_str(), "r"))
    {
        fclose(file);
        return true;
    }
    else
        return false;
}

// Owns one HDF5 identifier and closes it with the matching H5*close function
class qHDF_id
{
public:
    using closer_t = herr_t (*)(hid_t);
    qHDF_id() = default;
    qHDF_id(hid_t id, closer_t closer) : id_(id), closer_(closer) {}
    qHDF_id(const qHDF_id &) = delete;
    qHDF_id &operator=(const qHDF_id &) = delete;
    qHDF_id(qHDF_id &&other) noexcept : id_(other.id_), closer_(other.closer_) { other.id_ = H5I_INVALID_HID; }
    qHDF_id &operator=(qHDF_id &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            id_ = other.id_, closer_ = other.closer_;
            other.id_ = H5I_INVALID_HID;
        }
        return *this;
    }
    ~qHDF_id() { reset(); }
    void reset()
    {
        if (id_ >= 0 && closer_ != nullptr)
            closer_(id_);
        id_ = H5I_INVALID_HID;
    }
    herr_t close() // Close and report the result (for files that were written)
    {
        herr_t status = (id_ >= 0 && closer_ != nullptr) ? closer_(id_) : 0;
        id_ = H5I_INVALID_HID;
        return status;
    }
    bool valid() const { return id_ >= 0; }
    operator hid_t() const { return id_; }

private:
    hid_t id_ = H5I_INVALID_HID;
    closer_t closer_ = nullptr;
};

// Switches HDF5's automatic error printing off and restores the caller's error handler on destruction
class qHDF_error_guard
{
public:
    qHDF_error_guard()
    {
        saved_ = H5Eget_auto2(H5E_DEFAULT, &func_, &client_data_) >= 0;
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    }
    qHDF_error_guard(const qHDF_error_guard &) = delete;
    qHDF_error_guard &operator=(const qHDF_error_guard &) = delete;
    ~qHDF_error_guard()
    {
        if (saved_)
            H5Eset_auto2(H5E_DEFAULT, func_, client_data_);
    }

private:
    H5E_auto2_t func_ = nullptr;
    void *client_data_ = nullptr;
    bool saved_ = false;
};

// Throw if an HDF5 call failed while writing: "HDF5 error: <what>" or "HDF5 error: <what> '<name>'"
// (the message is only built on failure)
[[noreturn]] static void qHDF_write_error(const char *what, const char *name)
{
    std::string msg = std::string("HDF5 error: ") + what;
    if (name != nullptr)
        msg += std::string(" '") + name + "'";
    throw std::runtime_error(msg);
}

static inline void qHDF_check(herr_t status, const char *what, const char *name = nullptr)
{
    if (status < 0)
        qHDF_write_error(what, name);
}

static inline qHDF_id qHDF_check(qHDF_id id, const char *what, const char *name = nullptr)
{
    if (!id.valid())
        qHDF_write_error(what, name);
    return id;
}

// Helper function: convert to float
template <typename dtype>
static inline void qHDF_cast_to_float(const dtype *in, float *out, unsigned long long n_elem)
{
    for (auto n = 0ULL; n < n_elem; ++n)
        out[n] = float(in[n]);
}

// Helper function: check if string contains only alphanumeric characters
static bool qHDF_isalnum(const std::string &str)
{
    for (char ch : str)
        if (!std::isalnum((unsigned char)ch) && ch != '_')
            return false;
    return true;
}

// Native HDF5 memory type of a C++ type
template <typename T>
static inline hid_t qHDF_mem_type();
template <>
inline hid_t qHDF_mem_type<float>() { return H5T_NATIVE_FLOAT; }
template <>
inline hid_t qHDF_mem_type<double>() { return H5T_NATIVE_DOUBLE; }
template <>
inline hid_t qHDF_mem_type<unsigned long long>() { return H5T_NATIVE_ULLONG; }
template <>
inline hid_t qHDF_mem_type<long long>() { return H5T_NATIVE_LLONG; }
template <>
inline hid_t qHDF_mem_type<unsigned>() { return H5T_NATIVE_UINT; }
template <>
inline hid_t qHDF_mem_type<int>() { return H5T_NATIVE_INT; }

// Product of the extents with an overflow check; returns false on overflow
static inline bool qHDF_product(const hsize_t *dims, int rank, unsigned long long *n)
{
    unsigned long long p = 1ULL;
    for (int i = 0; i < rank; ++i)
        if (__builtin_mul_overflow(p, (unsigned long long)dims[i], &p))
            return false;
    *n = p;
    return true;
}

// A link with this (single-component) name exists in the group
static inline bool qHDF_link_exists(hid_t loc, const std::string &name)
{
    return H5Lexists(loc, name.c_str(), H5P_DEFAULT) > 0;
}

// Size of an open file in bytes; 0 if it cannot be determined (then every non-empty dataset fails the storage check)
static inline unsigned long long qHDF_file_size(hid_t file_id)
{
    hsize_t file_size = 0;
    return H5Fget_filesize(file_id, &file_size) < 0 ? 0ULL : (unsigned long long)file_size;
}

// Checks that a dataset holds data for every element it declares, so that a crafted file cannot make the reader
// allocate memory for data that is not there. Contiguous and compact datasets must have storage for every element;
// chunked datasets may be compressed, their storage is allowed to be smaller by up to a factor of 65536. The storage
// size is read from the file and is therefore also checked against the size of the file ('file_size', queried once
// per open file). The layout is only queried when the storage is smaller than the declared data.
static inline bool qHDF_storage_ok(hid_t dset_id, hid_t type_id, unsigned long long n_elem, unsigned long long file_size)
{
    if (n_elem == 0)
        return true;
    size_t type_size = H5Tget_size(type_id);
    unsigned long long n_bytes = 0;
    if (type_size == 0 || __builtin_mul_overflow(n_elem, (unsigned long long)type_size, &n_bytes))
        return false;

    // The storage size comes from the file itself; it cannot exceed the size of the file
    unsigned long long storage = (unsigned long long)H5Dget_storage_size(dset_id);
    if (storage > file_size)
        return false;
    if (storage >= n_bytes) // Enough for every layout
        return true;

    qHDF_id dcpl(H5Dget_create_plist(dset_id), H5Pclose);
    if (!dcpl.valid())
        return false;
    H5D_layout_t layout = H5Pget_layout(dcpl);
    if (layout == H5D_CONTIGUOUS || layout == H5D_COMPACT)
        return false;
    if (layout == H5D_CHUNKED)
        return storage != 0 && storage >= n_bytes / 65536ULL;
    return layout == H5D_VIRTUAL;
}

// Numeric dataset opened for reading: the rank and every extent (HDF5 order, slowest dimension first) are checked
// against 'expected' before the caller allocates any buffer; qHDF_ANY accepts any extent. Throws
// std::invalid_argument if the dataset is missing, has another shape, is not numeric, or cannot be read.
static constexpr unsigned long long qHDF_ANY = ~0ULL;
class qHDF_dataset
{
public:
    qHDF_dataset(hid_t loc, const char *name, std::initializer_list<unsigned long long> expected, unsigned long long file_size)
        : name_(name)
    {
        if (expected.size() > 3)
            throw std::invalid_argument("Internal error: rank > 3.");
        dset_ = qHDF_id(H5Dopen2(loc, name, H5P_DEFAULT), H5Dclose);
        if (!dset_.valid())
            fail("is missing or cannot be opened");
        qHDF_id space(H5Dget_space(dset_), H5Sclose);
        hsize_t dims[H5S_MAX_RANK];
        int rank = space.valid() ? H5Sget_simple_extent_dims(space, dims, nullptr) : -1; // Gives the rank as well
        if (rank != (int)expected.size())
        {
            if (rank < 0 || H5Sget_simple_extent_type(space) != H5S_SIMPLE)
                fail("has no simple dataspace");
            fail("has " + std::to_string(rank) + " dimensions, expected " + std::to_string(expected.size()));
        }
        int i = 0;
        for (auto e : expected)
        {
            dims_[i] = dims[i];
            if (e != qHDF_ANY && (unsigned long long)dims_[i] != e)
                fail("has extent " + std::to_string((unsigned long long)dims_[i]) + " in dimension " + std::to_string(i) +
                     ", expected " + std::to_string(e));
            ++i;
        }
        if (!qHDF_product(dims_, rank, &n_elem_))
            fail("is too large");
        qHDF_id type(H5Dget_type(dset_), H5Tclose);
        H5T_class_t type_class = type.valid() ? H5Tget_class(type) : H5T_NO_CLASS;
        if (type_class != H5T_FLOAT && type_class != H5T_INTEGER)
            fail("is not numeric");
        if (!qHDF_storage_ok(dset_, type, n_elem_, file_size))
            fail("declares more elements than the file contains");
        stored_as_float_ = H5Tequal(type, H5T_NATIVE_FLOAT) > 0;
    }
    unsigned long long dim(int i) const { return (unsigned long long)dims_[i]; }
    unsigned long long n_elem() const { return n_elem_; }
    template <typename T>
    void read(T *out) const // 'out' must hold n_elem() values; HDF5 converts the stored type to T
    {
        if (n_elem_ == 0)
            return;
        if constexpr (std::is_same<T, double>::value)
            if (stored_as_float_) // Single precision (as written by the library): read without HDF5 type conversion
            {
                std::unique_ptr<float[]> buffer(new float[(size_t)n_elem_]);
                if (H5Dread(dset_, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.get()) < 0)
                    fail("cannot be read");
                for (unsigned long long n = 0; n < n_elem_; ++n)
                    out[n] = (double)buffer[n];
                return;
            }
        if (H5Dread(dset_, qHDF_mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, out) < 0)
            fail("cannot be read");
    }

private:
    [[noreturn]] void fail(const std::string &what) const
    {
        throw std::invalid_argument("Malformed HDF5 file: dataset '" + std::string(name_) + "' " + what + ".");
    }
    const char *name_;
    qHDF_id dset_;
    hsize_t dims_[3] = {1, 1, 1};
    unsigned long long n_elem_ = 0;
    bool stored_as_float_ = false;
};

// Read a numeric attribute that holds exactly 'n' values. Returns false if it does not exist and is not required;
// throws std::invalid_argument if it is missing (and required), has another size, is not numeric, or cannot be read.
template <typename T>
static bool qHDF_read_attr(hid_t loc, const char *name, T *out, hssize_t n = 1, bool required = true)
{
    auto fail = [&](const std::string &what)
    { throw std::invalid_argument("Malformed HDF5 file: attribute '" + std::string(name) + "' " + what + "."); };
    qHDF_id attr(H5Aopen(loc, name, H5P_DEFAULT), H5Aclose);
    if (!attr.valid()) // Only on failure: tell a missing attribute from one that cannot be opened
    {
        if (H5Aexists(loc, name) <= 0)
        {
            if (required)
                fail("is missing");
            return false;
        }
        fail("cannot be opened");
    }
    // Fast path: a numeric type whose stored data has the size of exactly 'n' values (the stored size of an attribute
    // is the number of values times the type size). Otherwise the checks below run one by one for the message.
    qHDF_id type(H5Aget_type(attr), H5Tclose);
    H5T_class_t type_class = type.valid() ? H5Tget_class(type) : H5T_NO_CLASS;
    bool is_numeric = type_class == H5T_FLOAT || type_class == H5T_INTEGER;
    if (!is_numeric || (unsigned long long)H5Aget_storage_size(attr) != (unsigned long long)n * (unsigned long long)H5Tget_size(type))
    {
        qHDF_id space(H5Aget_space(attr), H5Sclose);
        if (!space.valid() || H5Sget_simple_extent_npoints(space) != n)
            fail("must have " + std::to_string(n) + (n == 1 ? " value" : " values"));
        if (!is_numeric)
            fail("is not numeric");
    }
    if (H5Aread(attr, qHDF_mem_type<T>(), out) < 0)
        fail("cannot be read");
    return true;
}

// Read the channel name: a fixed-length string of at most 256 bytes; an empty name if the attribute is missing
static std::string qHDF_read_name(hid_t loc)
{
    const std::string msg = "Malformed HDF5 file: attribute 'Name' ";
    qHDF_id attr(H5Aopen(loc, "Name", H5P_DEFAULT), H5Aclose);
    if (!attr.valid()) // Only on failure: tell a missing attribute from one that cannot be opened
    {
        if (H5Aexists(loc, "Name") <= 0)
            return std::string();
        throw std::invalid_argument(msg + "cannot be opened.");
    }
    qHDF_id type(H5Aget_type(attr), H5Tclose);
    qHDF_id space(H5Aget_space(attr), H5Sclose);
    if (!type.valid() || !space.valid())
        throw std::invalid_argument(msg + "cannot be opened.");
    if (H5Tget_class(type) != H5T_STRING || H5Tis_variable_str(type) != 0 || H5Tget_size(type) > 256 ||
        H5Sget_simple_extent_npoints(space) != 1)
        throw std::invalid_argument(msg + "must be a fixed-length string of at most 256 bytes.");
    char buffer[257] = {0}; // Zero-terminated even if the stored string fills all 256 bytes
    if (H5Aread(attr, type, buffer) < 0)
        throw std::invalid_argument(msg + "cannot be read.");
    return std::string(buffer);
}

// Create an attribute ('n' = 0: scalar, otherwise a vector of length n) and write its value
static void qHDF_create_attr(hid_t loc, const char *name, hid_t type, hsize_t n, const void *data)
{
    qHDF_id space = qHDF_check(qHDF_id(n == 0 ? H5Screate(H5S_SCALAR) : H5Screate_simple(1, &n, nullptr), H5Sclose),
                               "cannot create dataspace");
    qHDF_id attr = qHDF_check(qHDF_id(H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose),
                              "cannot create attribute", name);
    qHDF_check(H5Awrite(attr, type, data), "cannot write attribute", name);
}

// Write the value of an existing attribute
static void qHDF_write_attr(hid_t loc, const char *name, hid_t mem_type, const void *data)
{
    qHDF_id attr = qHDF_check(qHDF_id(H5Aopen(loc, name, H5P_DEFAULT), H5Aclose),
                              "cannot open attribute", name);
    qHDF_check(H5Awrite(attr, mem_type, data), "cannot write attribute", name);
}

// Create a numeric dataset of rank 1-3 ('dims' in HDF5 order, slowest dimension first) and write 'n_elem' values
static void qHDF_write_dset(hid_t loc, const char *name, hid_t file_type, hid_t mem_type,
                            int rank, const hsize_t *dims, const void *data, unsigned long long n_elem)
{
    qHDF_id space = qHDF_check(qHDF_id(rank == 0 ? H5Screate(H5S_SCALAR) : H5Screate_simple(rank, dims, nullptr), H5Sclose),
                               "cannot create dataspace");
    qHDF_id dset = qHDF_check(qHDF_id(H5Dcreate2(loc, name, file_type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose),
                              "cannot create dataset", name);
    if (n_elem != 0)
        qHDF_check(H5Dwrite(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "cannot write dataset", name);
}

// Write single-precision data from a float or a double array
template <typename dtype>
static void qHDF_write_float(hid_t loc, const char *name, int rank, const hsize_t *dims, const dtype *data, unsigned long long n_elem)
{
    if constexpr (std::is_same<dtype, float>::value)
        qHDF_write_dset(loc, name, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, rank, dims, data, n_elem);
    else
    {
        std::unique_ptr<float[]> buffer(new float[(size_t)n_elem]); // Not zeroed: every value is set below
        qHDF_cast_to_float(data, buffer.get(), n_elem);
        qHDF_write_dset(loc, name, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, rank, dims, buffer.get(), n_elem);
    }
}

// Storage index of a file: layout {nx, ny, nz, nw} and the channel ID of every slot (0 = empty slot)
struct qHDF_index
{
    unsigned dims[4] = {0, 0, 0, 0};
    std::vector<unsigned> order;
};

// Read the storage layout ('ChannelDims') and, if requested, the index ('Order') of an open file
static qHDF_index qHDF_read_index(hid_t file_id, unsigned long long file_size, bool read_order = true)
{
    qHDF_index idx;
    try
    {
        qHDF_dataset channel_dims(file_id, "ChannelDims", {4}, file_size);
        channel_dims.read(idx.dims);
        if (!read_order)
            return idx;

        // Number of slots, at most 2^32 - 1 (each step stays below 2^64)
        unsigned long long n_order = 1ULL;
        for (unsigned d : idx.dims)
            if ((n_order *= (unsigned long long)d) > (unsigned long long)UINT_MAX)
                throw std::invalid_argument("");

        qHDF_dataset order(file_id, "Order", {n_order}, file_size);
        idx.order.resize((size_t)n_order);
        order.read(idx.order.data());
    }
    catch (const std::invalid_argument &)
    {
        throw std::invalid_argument("Storage index in HDF file is corrupted.");
    }
    return idx;
}

// Open an existing HDF5 file
static qHDF_id qHDF_open_file(const std::string &fn, unsigned flags, const char *error_message = "Error opening file.")
{
    if (H5Fis_hdf5(fn.c_str()) <= 0)
        throw std::invalid_argument("Not an HDF5 file.");
    qHDF_id file(H5Fopen(fn.c_str(), flags, H5P_DEFAULT), H5Fclose);
    if (!file.valid())
        throw std::invalid_argument(error_message);
    return file;
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

// Linear slot number of (ix, iy, iz, iw), or -1 if the index is outside the layout
static long long qHDF_slot(const qHDF_index &idx, unsigned ix, unsigned iy, unsigned iz, unsigned iw)
{
    if (ix >= idx.dims[0] || iy >= idx.dims[1] || iz >= idx.dims[2] || iw >= idx.dims[3])
        return -1LL;
    unsigned long long slot = (((unsigned long long)iw * idx.dims[2] + iz) * idx.dims[1] + iy) * idx.dims[0] + ix;
    if (slot >= (unsigned long long)idx.order.size())
        return -1LL;
    return (long long)slot;
}

// Channel ID stored at a slot; 0 if the slot is empty or outside the layout
static unsigned qHDF_channel_id(hid_t file_id, unsigned long long file_size, unsigned ix, unsigned iy, unsigned iz, unsigned iw)
{
    qHDF_index idx = qHDF_read_index(file_id, file_size);
    long long slot = qHDF_slot(idx, ix, iy, iz, iw);
    return slot < 0 ? 0 : idx.order[(size_t)slot];
}

static inline std::string qHDF_group_name(unsigned channel_id)
{
    return "/channel_" + std::to_string(channel_id);
}

// Open the group of a channel ID taken from the index
static qHDF_id qHDF_open_channel_group(hid_t file_id, unsigned channel_id)
{
    qHDF_id group(H5Gopen2(file_id, qHDF_group_name(channel_id).c_str(), H5P_DEFAULT), H5Gclose);
    if (!group.valid())
        throw std::invalid_argument("Storage index in HDF file is corrupted.");
    return group;
}

// Write the storage index (the extent was checked by qHDF_read_index on the same open file)
static void qHDF_write_order(hid_t file_id, const std::vector<unsigned> &order)
{
    qHDF_id dset = qHDF_check(qHDF_id(H5Dopen2(file_id, "Order", H5P_DEFAULT), H5Dclose), "cannot open the storage index");
    qHDF_check(H5Dwrite(dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, order.data()), "cannot write the storage index");
}

// A new channel ID: one above the largest ID in the index, skipping group names that are already taken
static unsigned qHDF_new_channel_id(hid_t file_id, const std::vector<unsigned> &order)
{
    unsigned long long id = 1ULL;
    for (unsigned v : order)
        if ((unsigned long long)v >= id)
            id = (unsigned long long)v + 1ULL;
    while (id <= (unsigned long long)UINT_MAX && qHDF_link_exists(file_id, "channel_" + std::to_string(id)))
        ++id;
    if (id > (unsigned long long)UINT_MAX)
        throw std::runtime_error("No free channel ID left in the HDF5 file.");
    return (unsigned)id;
}

// Create the group of a new channel with the default attributes of an empty channel
static qHDF_id qHDF_create_channel_group(hid_t file_id, unsigned channel_id)
{
    const std::string name = qHDF_group_name(channel_id);
    qHDF_id group = qHDF_check(qHDF_id(H5Gcreate2(file_id, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose),
                               "cannot create group", name.c_str());
    unsigned zero = 0;
    int zero_int = 0;
    float nan3[3] = {NAN, NAN, NAN}, zero3[3] = {0.0f, 0.0f, 0.0f};
    qHDF_create_attr(group, "NumSnap", H5T_NATIVE_UINT, 0, &zero);
    qHDF_create_attr(group, "NumTx", H5T_NATIVE_UINT, 0, &zero);
    qHDF_create_attr(group, "NumRx", H5T_NATIVE_UINT, 0, &zero);
    qHDF_create_attr(group, "Initial_position", H5T_NATIVE_INT, 0, &zero_int);
    qHDF_create_attr(group, "CenterFrequency", H5T_NATIVE_FLOAT, 0, nan3); // NaN = not set
    qHDF_create_attr(group, "tx_position", H5T_NATIVE_FLOAT, 3, nan3);
    qHDF_create_attr(group, "rx_position", H5T_NATIVE_FLOAT, 3, nan3);
    qHDF_create_attr(group, "tx_orientation", H5T_NATIVE_FLOAT, 3, zero3);
    qHDF_create_attr(group, "rx_orientation", H5T_NATIVE_FLOAT, 3, zero3);

    // Name: fixed-length string of 256 bytes, NUL-terminated
    char name_buffer[256] = {0};
    qHDF_id type = qHDF_check(qHDF_id(H5Tcopy(H5T_C_S1), H5Tclose), "cannot create string type");
    qHDF_check(H5Tset_size(type, 256), "cannot set string size");
    qHDF_check(H5Tset_strpad(type, H5T_STR_NULLTERM), "cannot set string padding");
    qHDF_create_attr(group, "Name", type, 0, name_buffer);
    return group;
}

// Create the group for unstructured data; it tracks the creation order, so datasets are listed as written
static qHDF_id qHDF_create_data_group(hid_t group_id)
{
    qHDF_id gcpl = qHDF_check(qHDF_id(H5Pcreate(H5P_GROUP_CREATE), H5Pclose), "cannot create property list");
    qHDF_check(H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED), "cannot set creation order");
    return qHDF_check(qHDF_id(H5Gcreate2(group_id, "Data", H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose), "cannot create group 'Data'");
}

// Open the group for unstructured data
static qHDF_id qHDF_open_data_group(hid_t group_id)
{
    qHDF_id data(H5Gopen2(group_id, "Data", H5P_DEFAULT), H5Gclose);
    if (!data.valid())
        throw std::invalid_argument("Malformed HDF5 file: group 'Data' cannot be opened.");
    return data;
}

// Deletes the link to a newly created group unless released: a group that was not written completely is removed
class qHDF_unlink_guard
{
public:
    qHDF_unlink_guard(hid_t loc, std::string name, bool active) : loc_(loc), name_(std::move(name)), active_(active) {}
    qHDF_unlink_guard(const qHDF_unlink_guard &) = delete;
    qHDF_unlink_guard &operator=(const qHDF_unlink_guard &) = delete;
    ~qHDF_unlink_guard()
    {
        if (active_)
            H5Ldelete(loc_, name_.c_str(), H5P_DEFAULT);
    }
    void release() { active_ = false; }

private:
    hid_t loc_;
    std::string name_;
    bool active_;
};

// HDF5 type of a supported numeric unstructured data type (type IDs of quadriga_lib::any_type_id)
static hid_t qHDF_par_type(int datatype_id)
{
    if (datatype_id < 10 || datatype_id > 55 || datatype_id % 10 > 5)
        return H5I_INVALID_HID;
    switch (datatype_id % 10)
    {
    case 0:
        return H5T_NATIVE_FLOAT;
    case 1:
        return H5T_NATIVE_DOUBLE;
    case 2:
        return H5T_NATIVE_ULLONG;
    case 3:
        return H5T_NATIVE_LLONG;
    case 4:
        return H5T_NATIVE_UINT;
    default:
        return H5T_NATIVE_INT;
    }
}

// Check the name (including the prefix) and the type of an unstructured dataset before anything is written
static void qHDF_check_par(const std::string &par_name, const std::any *par_data)
{
    if (par_name.length() == 0 || !qHDF_isalnum(par_name))
        throw std::invalid_argument("Parameter name must only contain letters, numbers and the underscore '_'.");
    if (par_data == nullptr)
        throw std::invalid_argument("NULL pointer was passed as data object.");
    int datatype_id = quadriga_lib::any_type_id(par_data);
    if (datatype_id != 9 && qHDF_par_type(datatype_id) == H5I_INVALID_HID)
        throw std::invalid_argument("Unsupported data type for '" + par_name + "'");
}

// Write one unstructured dataset
// - Scalars get a scalar dataspace, vectors rank 1 (also with a single element), matrices rank 2, cubes rank 3
// - Strings are stored as fixed-length strings; an empty string as one NUL byte (HDF5 has no strings of size 0)
static void qHDF_write_par(hid_t group_id, const std::string &par_name, const std::any *par_data)
{
    qHDF_check_par(par_name, par_data);
    if (qHDF_link_exists(group_id, par_name))
        throw std::invalid_argument("Dataset '" + par_name + "' already exists.");

    unsigned long long dims_arma[3];
    void *dataptr = nullptr;
    int datatype_id = quadriga_lib::any_type_id(par_data, dims_arma, &dataptr);

    if (datatype_id == 9)
    {
        const std::string *str = std::any_cast<std::string>(par_data);
        qHDF_id type = qHDF_check(qHDF_id(H5Tcopy(H5T_C_S1), H5Tclose), "cannot create string type");
        qHDF_check(H5Tset_size(type, std::max(str->size(), (size_t)1)), "cannot set string size");
        qHDF_write_dset(group_id, par_name.c_str(), type, type, 0, nullptr, str->c_str(), 1);
        return;
    }

    hid_t type = qHDF_par_type(datatype_id);
    hsize_t dims[3] = {1, 1, 1};
    if (datatype_id < 20) // Scalars
        qHDF_write_dset(group_id, par_name.c_str(), type, type, 0, dims, dataptr, 1);
    else if (datatype_id < 30) // Matrices
    {
        dims[0] = (hsize_t)dims_arma[1], dims[1] = (hsize_t)dims_arma[0];
        qHDF_write_dset(group_id, par_name.c_str(), type, type, 2, dims, dataptr, dims_arma[0] * dims_arma[1]);
    }
    else if (datatype_id < 40) // Cubes
    {
        dims[0] = (hsize_t)dims_arma[2], dims[1] = (hsize_t)dims_arma[1], dims[2] = (hsize_t)dims_arma[0];
        qHDF_write_dset(group_id, par_name.c_str(), type, type, 3, dims, dataptr, dims_arma[0] * dims_arma[1] * dims_arma[2]);
    }
    else // Column and row vectors
    {
        dims[0] = (hsize_t)dims_arma[0];
        qHDF_write_dset(group_id, par_name.c_str(), type, type, 1, dims, dataptr, dims_arma[0]);
    }
}

// Callback of H5Literate: collect the link names of a group
static herr_t qHDF_collect_link_name(hid_t, const char *name, const H5L_info_t *, void *names)
{
    try
    {
        static_cast<std::vector<std::string> *>(names)->emplace_back(name);
        return 0;
    }
    catch (...)
    {
        return -1;
    }
}

// Names of the unstructured datasets in a group that start with 'prefix' (returned without the prefix), appended to
// 'par_names'; in creation order if the group tracks it (groups written by this library do), otherwise in name order.
// With 'datasets_only = false', links to other objects are listed too (the caller then skips them when it fails to
// open them as datasets). Returns the number of names found.
static unsigned long long qHDF_read_par_names(hid_t group_id, std::vector<std::string> *par_names, const std::string &prefix,
                                              bool datasets_only = true)
{
    // All names in one pass (names of any length); a group that does not track the creation order makes the
    // first H5Literate fail
    std::vector<std::string> names;
    hsize_t position = 0;
    if (H5Literate(group_id, H5_INDEX_CRT_ORDER, H5_ITER_INC, &position, qHDF_collect_link_name, &names) < 0)
    {
        names.clear(), position = 0;
        if (H5Literate(group_id, H5_INDEX_NAME, H5_ITER_INC, &position, qHDF_collect_link_name, &names) < 0)
            throw std::invalid_argument("Failed to get group info.");
    }

    unsigned long long n_found = 0;
    for (const std::string &name : names)
    {
        if (name.compare(0, prefix.length(), prefix) != 0)
            continue;

        if (datasets_only) // Only datasets are listed
        {
            qHDF_id object(H5Oopen(group_id, name.c_str(), H5P_DEFAULT), H5Oclose);
            if (!object.valid() || H5Iget_type(object) != H5I_DATASET)
                continue;
        }

        if (par_names != nullptr)
            par_names->push_back(name.substr(prefix.length()));
        ++n_found;
    }
    return n_found;
}

// Read a numeric unstructured dataset of rank 0 (scalar) to 3 with element type T
template <typename T>
static std::any qHDF_read_numeric(hid_t dset_id, int rank, const hsize_t *dims, const std::string &name)
{
    auto read = [&](T *out, unsigned long long n_elem)
    {
        if (n_elem != 0 && H5Dread(dset_id, qHDF_mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, out) < 0)
            throw std::invalid_argument("Malformed HDF5 file: unstructured dataset '" + name + "' cannot be read.");
    };
    if (rank == 0)
    {
        T value{};
        read(&value, 1);
        return value;
    }
    if (rank == 1)
    {
        arma::Col<T> value((arma::uword)dims[0], arma::fill::none);
        read(value.memptr(), value.n_elem);
        return value;
    }
    if (rank == 2)
    {
        arma::Mat<T> value((arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
        read(value.memptr(), value.n_elem);
        return value;
    }
    arma::Cube<T> value((arma::uword)dims[2], (arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
    read(value.memptr(), value.n_elem);
    return value;
}

// Read one unstructured dataset; returns an empty std::any if it does not exist or has an unsupported type or shape
// - A scalar dataspace gives a scalar, rank 1 an arma::Col (also with one element), rank 2 an arma::Mat, rank 3 an
//   arma::Cube; 32- and 64-bit integers and 32- and 64-bit floats are supported, rank > 3 is not
// - Fixed-length strings are returned without trailing NUL bytes; variable-length strings (h5py, MATLAB) are copied
//   and their buffer is freed
// - Throws std::invalid_argument if a supported dataset cannot be read
static std::any qHDF_read_data(hid_t group_id, const std::string &dataset_name, unsigned long long file_size,
                               bool check_exists = true)
{
    if (check_exists && !qHDF_link_exists(group_id, dataset_name)) // Not needed for names just listed
        return std::any();
    qHDF_id dset(H5Dopen2(group_id, dataset_name.c_str(), H5P_DEFAULT), H5Dclose);
    if (!dset.valid()) // Not a dataset
        return std::any();

    auto msg = [&](const char *what) { return "Malformed HDF5 file: unstructured dataset '" + dataset_name + "' " + what; };
    qHDF_id type(H5Dget_type(dset), H5Tclose), space(H5Dget_space(dset), H5Sclose);
    if (!type.valid() || !space.valid())
        throw std::invalid_argument(msg("cannot be opened."));

    int rank = 0;
    hsize_t dims[3] = {1, 1, 1};
    H5S_class_t space_class = H5Sget_simple_extent_type(space);
    if (space_class == H5S_SIMPLE)
    {
        rank = H5Sget_simple_extent_ndims(space);
        if (rank < 1 || rank > 3)
            return std::any(); // Unsupported shape
        if (H5Sget_simple_extent_dims(space, dims, nullptr) < 0)
            throw std::invalid_argument(msg("has unreadable extents."));
    }
    else if (space_class != H5S_SCALAR)
        return std::any();

    unsigned long long n_elem = 1ULL;
    if (!qHDF_product(dims, rank, &n_elem))
        throw std::invalid_argument(msg("is too large."));

    H5T_class_t type_class = H5Tget_class(type);
    if (type_class == H5T_STRING)
    {
        if (n_elem != 1)
            return std::any(); // Arrays of strings are not supported

        htri_t is_variable = H5Tis_variable_str(type);
        if (is_variable < 0)
            throw std::invalid_argument(msg("has an unreadable string type."));
        if (is_variable > 0)
        {
            qHDF_id mem_type(H5Tcopy(H5T_C_S1), H5Tclose);
            if (!mem_type.valid() || H5Tset_size(mem_type, H5T_VARIABLE) < 0 || H5Tset_cset(mem_type, H5Tget_cset(type)) < 0)
                throw std::invalid_argument(msg("has an unsupported string type."));
            char *ptr = nullptr;
            if (H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ptr) < 0)
                throw std::invalid_argument(msg("cannot be read."));
            std::unique_ptr<char, herr_t (*)(void *)> owner(ptr, H5free_memory);
            return ptr == nullptr ? std::string() : std::string(ptr);
        }

        size_t size = H5Tget_size(type);
        if (size == 0 || !qHDF_storage_ok(dset, type, 1, file_size))
            throw std::invalid_argument(msg("declares more data than the file contains."));
        std::string value(size, '\0');
        if (H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value[0]) < 0)
            throw std::invalid_argument(msg("cannot be read."));
        while (!value.empty() && value.back() == '\0')
            value.pop_back();
        return value;
    }

    // Numeric types: 0 = float, 1 = double, 2 = uint64, 3 = int64, 4 = uint32, 5 = int32 (as in any_type_id)
    size_t size = H5Tget_size(type);
    int kind = -1;
    if (type_class == H5T_FLOAT)
        kind = (size == 4) ? 0 : (size == 8) ? 1
                                             : -1;
    else if (type_class == H5T_INTEGER)
    {
        bool is_unsigned = H5Tget_sign(type) == H5T_SGN_NONE;
        kind = (size == 8) ? (is_unsigned ? 2 : 3) : (size == 4) ? (is_unsigned ? 4 : 5)
                                                                 : -1;
    }
    if (kind < 0)
        return std::any(); // Unsupported type

    if (!qHDF_storage_ok(dset, type, n_elem, file_size))
        throw std::invalid_argument(msg("declares more elements than the file contains."));

    switch (kind)
    {
    case 0:
        return qHDF_read_numeric<float>(dset, rank, dims, dataset_name);
    case 1:
        return qHDF_read_numeric<double>(dset, rank, dims, dataset_name);
    case 2:
        return qHDF_read_numeric<unsigned long long>(dset, rank, dims, dataset_name);
    case 3:
        return qHDF_read_numeric<long long>(dset, rank, dims, dataset_name);
    case 4:
        return qHDF_read_numeric<unsigned>(dset, rank, dims, dataset_name);
    default:
        return qHDF_read_numeric<int>(dset, rank, dims, dataset_name);
    }
}

// Indices of the snapshot groups 'Snap_<i>' (i < n_snapshots) of a channel group, sorted
static std::vector<unsigned> qHDF_snapshot_groups(hid_t group_id, unsigned n_snapshots)
{
    // All link names in one pass
    std::vector<std::string> names;
    hsize_t position = 0;
    if (H5Literate(group_id, H5_INDEX_NAME, H5_ITER_INC, &position, qHDF_collect_link_name, &names) < 0)
        throw std::invalid_argument("Malformed HDF5 file: the channel group cannot be listed.");

    std::vector<unsigned> snaps;
    for (const std::string &name : names)
    {
        if (name.size() < 6 || name.size() > 15 || name.compare(0, 5, "Snap_") != 0) // "Snap_" and 1 to 10 digits
            continue;
        unsigned long long value = 0;
        bool is_number = true;
        for (size_t p = 5; p < name.size(); ++p)
            if (name[p] < '0' || name[p] > '9')
                is_number = false;
            else
                value = value * 10ULL + (unsigned long long)(name[p] - '0');
        if (is_number && value < (unsigned long long)n_snapshots && std::to_string(value) == name.substr(5))
            snaps.push_back((unsigned)value);
    }
    std::sort(snaps.begin(), snaps.end());
    return snaps;
}

// Files written before the round-trip fix omit the groups of empty snapshots; the reader pads them with empty
// entries. A file needing more padding than this is rejected (it would only allocate memory for nothing).
static constexpr unsigned long long qHDF_max_missing_snapshots = 1ULL << 20;

// Write the structured data of a channel into its (new) group
template <typename dtype>
static void qHDF_write_channel_data(hid_t group_id, const quadriga_lib::channel<dtype> *ch)
{
    const arma::uword n_snap_arma = ch->n_snap();
    const size_t n_snap = (size_t)n_snap_arma;
    if (n_snap_arma > (arma::uword)UINT_MAX)
        throw std::invalid_argument("Too many snapshots.");
    const arma::uvec n_paths = ch->n_path();
    if (n_paths.n_elem != n_snap_arma)
        throw std::invalid_argument("Number of elements returned by 'n_path()' does not match number of snapshots.");

    // Channel name: copied into a zeroed 256-byte buffer, at most 255 characters, always NUL-terminated
    {
        char name_buffer[256] = {0};
        std::memcpy(name_buffer, ch->name.data(), std::min(ch->name.size(), (size_t)255));
        qHDF_id type = qHDF_check(qHDF_id(H5Tcopy(H5T_C_S1), H5Tclose), "cannot create string type");
        qHDF_check(H5Tset_size(type, 256), "cannot set string size");
        qHDF_check(H5Tset_strpad(type, H5T_STR_NULLTERM), "cannot set string padding");
        qHDF_write_attr(group_id, "Name", type, name_buffer);
    }

    // Number of snapshots, transmitters, receivers; index of the reference position
    unsigned n_snap_u = (unsigned)n_snap_arma, n_tx = (unsigned)ch->n_tx(), n_rx = (unsigned)ch->n_rx();
    qHDF_write_attr(group_id, "NumSnap", H5T_NATIVE_UINT, &n_snap_u);
    qHDF_write_attr(group_id, "NumTx", H5T_NATIVE_UINT, &n_tx);
    qHDF_write_attr(group_id, "NumRx", H5T_NATIVE_UINT, &n_rx);
    qHDF_write_attr(group_id, "Initial_position", H5T_NATIVE_INT, &ch->initial_position);

    // Center frequency in [Hz]: attribute if the same for all snapshots, dataset if one per snapshot; if empty, the
    // attribute keeps its default NaN, which the reader returns as an empty vector
    if (ch->center_frequency.n_elem == 1ULL)
    {
        float value = (float)ch->center_frequency.at(0);
        qHDF_write_attr(group_id, "CenterFrequency", H5T_NATIVE_FLOAT, &value);
    }
    else if (ch->center_frequency.n_elem == n_snap_arma && n_snap != 0)
    {
        hsize_t dims = (hsize_t)n_snap;
        qHDF_write_float(group_id, "CenterFrequency", 1, &dims, ch->center_frequency.memptr(), n_snap);
    }

    // Positions and orientations, matrix of size [3, n_snap] (dataset) or [3, 1] (attribute); empty: default kept
    auto write_position = [&](const arma::Mat<dtype> &value, const char *name, const char *field)
    {
        if (value.n_elem == 0)
            return;
        if (value.n_rows != 3)
            throw std::invalid_argument("'" + std::string(field) + "' must have 3 rows.");
        if (value.n_cols == 1)
        {
            float data[3] = {(float)value.at(0), (float)value.at(1), (float)value.at(2)};
            qHDF_write_attr(group_id, name, H5T_NATIVE_FLOAT, data);
        }
        else if (value.n_cols == n_snap_arma && n_snap != 0)
        {
            hsize_t dims[2] = {(hsize_t)value.n_cols, 3};
            qHDF_write_float(group_id, name, 2, dims, value.memptr(), value.n_elem);
        }
    };
    write_position(ch->tx_pos, "tx_position", "tx_pos");
    write_position(ch->rx_pos, "rx_position", "rx_pos");
    write_position(ch->tx_orientation, "tx_orientation", "tx_orientation");
    write_position(ch->rx_orientation, "rx_orientation", "rx_orientation");

    // Snapshot data: a field is written for every snapshot if it is present (one entry per snapshot), also for
    // snapshots without paths (datasets with extent 0), so that the reader restores every entry with its shape
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

    if (!(write_coeff || write_delay || write_path_gain || write_path_length || write_path_polarization ||
          write_path_angles || write_path_fbs_pos || write_path_lbs_pos || write_no_interact || write_interact_coord))
        return;

    auto write_cube = [](hid_t loc, const char *name, const arma::Cube<dtype> &value)
    {
        hsize_t dims[3] = {(hsize_t)value.n_slices, (hsize_t)value.n_cols, (hsize_t)value.n_rows};
        qHDF_write_float(loc, name, 3, dims, value.memptr(), value.n_elem);
    };
    auto write_mat = [](hid_t loc, const char *name, const arma::Mat<dtype> &value)
    {
        hsize_t dims[2] = {(hsize_t)value.n_cols, (hsize_t)value.n_rows};
        qHDF_write_float(loc, name, 2, dims, value.memptr(), value.n_elem);
    };
    auto write_col = [](hid_t loc, const char *name, const arma::Col<dtype> &value)
    {
        hsize_t dims = (hsize_t)value.n_elem;
        qHDF_write_float(loc, name, 1, &dims, value.memptr(), value.n_elem);
    };

    for (size_t i = 0; i < n_snap; ++i)
    {
        // One snapshot: data in the channel group; several: one group per snapshot
        qHDF_id snap_group;
        hid_t snap_id = group_id;
        if (n_snap != 1)
        {
            const std::string snap_name = "Snap_" + std::to_string(i);
            snap_group = qHDF_check(qHDF_id(H5Gcreate2(group_id, snap_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose),
                                    "cannot create group", snap_name.c_str());
            snap_id = snap_group;
        }

        unsigned n_pth = (unsigned)n_paths.at(i);
        qHDF_create_attr(snap_id, "NumPath", H5T_NATIVE_UINT, 0, &n_pth);

        // Channel coefficients, tensors of size [n_rx, n_tx, n_path]
        if (write_coeff)
        {
            if (arma::size(ch->coeff_re[i]) != arma::size(ch->coeff_im[i]))
                throw std::invalid_argument("Size mismatch between 'coeff_re[" + std::to_string(i) + "]' and 'coeff_im[" + std::to_string(i) + "]'.");
            write_cube(snap_id, "coeff_re", ch->coeff_re[i]);
            write_cube(snap_id, "coeff_im", ch->coeff_im[i]);
        }

        // Path delays in seconds, tensors of size [n_rx, n_tx, n_path] or [1, 1, n_path]
        if (write_delay)
            write_cube(snap_id, "delay", ch->delay[i]);

        // Path gain before antenna patterns, vectors of length [n_path]
        if (write_path_gain)
            write_col(snap_id, "path_gain", ch->path_gain[i]);

        // Absolute path length from TX to RX phase center, vectors of length [n_path]
        if (write_path_length)
            write_col(snap_id, "path_length", ch->path_length[i]);

        // Polarization transfer function, matrices of size [8, n_path], interleaved complex
        if (write_path_polarization)
            write_mat(snap_id, "path_polarization", ch->path_polarization[i]);

        // Departure and arrival angles, matrices of size [n_path, 4], {AOD, EOD, AOA, EOA}
        if (write_path_angles)
            write_mat(snap_id, "path_angles", ch->path_angles[i]);

        // First-bounce and last-bounce scatterer positions, matrices of size [3, n_path]
        if (write_path_fbs_pos)
            write_mat(snap_id, "path_fbs_pos", ch->path_fbs_pos[i]);
        if (write_path_lbs_pos)
            write_mat(snap_id, "path_lbs_pos", ch->path_lbs_pos[i]);

        // Number of interaction points of a path with the environment, 0 = LOS, vectors of length [n_path]
        if (write_no_interact)
        {
            hsize_t dims = (hsize_t)ch->no_interact[i].n_elem;
            qHDF_write_dset(snap_id, "no_interact", H5T_NATIVE_UINT, H5T_NATIVE_UINT, 1, &dims,
                            ch->no_interact[i].memptr(), ch->no_interact[i].n_elem);
        }

        // Interaction coordinates, matrices of size [3, sum(no_interact)]
        if (write_interact_coord)
            write_mat(snap_id, "interact_coord", ch->interact_coord[i]);
    }
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
    const unsigned long long n_order = qHDF_slot_count(nx, ny, nz, nw);

    qHDF_error_guard error_guard;

    // Create and open file
    qHDF_id file(H5Fcreate(fn.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    if (!file.valid()) // May be invalid if file is currently opened
        throw std::invalid_argument("Error creating file. It may be already opened.");

    // Store dimension size to file
    unsigned ChannelDims[4] = {nx, ny, nz, nw};
    hsize_t dims = 4;
    qHDF_write_dset(file, "ChannelDims", H5T_NATIVE_UINT32, H5T_NATIVE_UINT32, 1, &dims, ChannelDims, 4);

    // Create the index (all slots empty)
    std::vector<unsigned> order((size_t)n_order, 0U);
    dims = (hsize_t)n_order;
    qHDF_write_dset(file, "Order", H5T_NATIVE_UINT32, H5T_NATIVE_UINT32, 1, &dims, order.data(), n_order);

    // Write a version ID
    unsigned version = 2;
    qHDF_write_dset(file, "Version", H5T_NATIVE_UINT, H5T_NATIVE_UINT, 0, nullptr, &version, 1);

    qHDF_check(file.close(), "cannot close file", fn.c_str());
}

/*!MD
# hdf5_read_layout
Read the storage layout of an HDF5 channel file

- Returns `{nx, ny, nz, nw}` describing the 4D slot grid of the file.
- Returns `{0, 0, 0, 0}` if the file does not exist; throws if the file exists but is not a valid HDF5 file.
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
    if (!qHDF_file_exists(fn))
        return arma::u32_vec(4, arma::fill::zeros);

    qHDF_error_guard error_guard;
    qHDF_id file = qHDF_open_file(fn, H5F_ACC_RDONLY);
    qHDF_index idx = qHDF_read_index(file, qHDF_file_size(file), channelID != nullptr);

    arma::u32_vec ChannelDims = {idx.dims[0], idx.dims[1], idx.dims[2], idx.dims[3]};
    if (channelID != nullptr)
    {
        channelID->set_size((arma::uword)idx.order.size());
        std::copy(idx.order.begin(), idx.order.end(), channelID->begin());
    }
    return ChannelDims;
}

/*!MD
# hdf5_write
Write a channel object to an HDF5 file at a specified 4D index

- Writes a `quadriga_lib::channel<dtype>` object to slot `(ix, iy, iz, iw)` in the HDF5 file.
- Creates the file with default layout `(65536 × 1 × 1 × 1)` if it does not exist; appends to existing files.
- If the slot already holds data, the slot is replaced completely: the old structured data and all unstructured
  datasets of the slot (`par_*`, including those added with [[hdf5_write_dset]]) are removed, and only the new
  channel is stored there. HDF5 does not reuse the space of the removed data: the file does not shrink; use
  `h5repack` to reclaim the space.
- Throws if the index was not reserved during [[hdf5_create]].
- Structured data is always stored in single precision; input may be float or double.
- Unstructured data: supported types are string, double, float, (u)int32, (u)int64; up to 3D; storage order preserved.
- Set `assume_valid = true` to skip integrity validation (faster but unsafe for potentially corrupted data).
- The channel name is stored with at most 255 characters.

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
    for (size_t i = 0; i < ch->par_names.size(); ++i)
        qHDF_check_par("par_" + ch->par_names[i], &ch->par_data[i]);

    qHDF_error_guard error_guard;

    // Create file
    if (!qHDF_file_exists(fn))
        quadriga_lib::hdf5_create(fn);

    // Open file for writing
    qHDF_id file = qHDF_open_file(fn, H5F_ACC_RDWR, "Error opening file. It may be opened by another program.");

    // Slot and its current channel ID (0 = empty)
    qHDF_index idx = qHDF_read_index(file, qHDF_file_size(file));
    const long long slot = qHDF_slot(idx, ix, iy, iz, iw);
    if (slot < 0)
        throw std::invalid_argument("Index out of bound.");
    const unsigned old_channel_index = idx.order[(size_t)slot];

    // The slot is replaced completely: the channel is written to a new group, then the index is updated and the link
    // to the old group of the slot is deleted. If writing fails, the new group is removed and the slot is unchanged.
    const unsigned channel_index = qHDF_new_channel_id(file, idx.order);
    qHDF_unlink_guard remove_on_error(file, qHDF_group_name(channel_index), true);
    {
        qHDF_id group = qHDF_create_channel_group(file, channel_index);

        if (!channel_is_empty)
            qHDF_write_channel_data(group, ch);

        // Write unstructured data
        if (!ch->par_names.empty())
        {
            qHDF_id data = qHDF_create_data_group(group);
            for (size_t i = 0; i < ch->par_names.size(); ++i)
                qHDF_write_par(data, "par_" + ch->par_names[i], &ch->par_data[i]);
        }
    }
    idx.order[(size_t)slot] = channel_index;
    qHDF_write_order(file, idx.order);
    remove_on_error.release();

    // Delete the link to the old group (HDF5 reclaims the space only when the file is repacked)
    if (old_channel_index != 0 && std::count(idx.order.begin(), idx.order.end(), old_channel_index) == 0 &&
        qHDF_link_exists(file, "channel_" + std::to_string(old_channel_index)))
        qHDF_check(H5Ldelete(file, qHDF_group_name(old_channel_index).c_str(), H5P_DEFAULT),
                   "cannot delete the old data of the slot");

    qHDF_check(file.close(), "cannot close file", fn.c_str());
    return old_channel_index != 0 ? 1 : 0;
}

template int quadriga_lib::hdf5_write(const quadriga_lib::channel<float> *ch, std::string fn,
                                      unsigned ix, unsigned iy, unsigned iz, unsigned iw, bool assume_valid);

template int quadriga_lib::hdf5_write(const quadriga_lib::channel<double> *ch, std::string fn,
                                      unsigned ix, unsigned iy, unsigned iz, unsigned iw, bool assume_valid);

/*!MD
# hdf5_read_channel
Read a channel object from an HDF5 file at a specified 4D index

- Returns an empty channel object (`no_snapshots == 0`) if the slot contains no valid data.
- Structured data is stored in single precision in the file and converted to `dtype` on read.
- Unstructured fields (`std::any`) retain their original stored type without conversion; see [[hdf5_read_dset]]
  for the returned types.
- Round trip with [[hdf5_write]]: every per-snapshot field comes back with one entry per snapshot, also for
  snapshots without paths (empty entries keep their shape); a channel written without `center_frequency` comes back
  with an empty `center_frequency`; empty `tx_orientation` / `rx_orientation` come back as `[3, 1]` zeros.
- Throws `std::invalid_argument` if the file is malformed, e.g. if a dataset does not have the size given by the
  attributes of the channel, or a required dataset or attribute is missing.

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
    if (!qHDF_file_exists(fn))
        throw std::invalid_argument("File does not exist.");

    qHDF_error_guard error_guard;

    // Open file for reading
    qHDF_id file = qHDF_open_file(fn, H5F_ACC_RDONLY);

    // Get channel ID
    const unsigned long long file_size = qHDF_file_size(file);
    const unsigned channel_index = qHDF_channel_id(file, file_size, ix, iy, iz, iw);
    if (channel_index == 0)
        return quadriga_lib::channel<dtype>();

    // Open group
    qHDF_id group = qHDF_open_channel_group(file, channel_index);
    quadriga_lib::channel<dtype> c;

    // Read unstructured data
    if (qHDF_link_exists(group, "Data"))
    {
        const std::string prefix = "par_";
        std::vector<std::string> available_par;
        qHDF_id data = qHDF_open_data_group(group);
        qHDF_read_par_names(data, &available_par, prefix, false); // Non-datasets give an empty std::any below
        for (std::string &name : available_par)
        {
            std::any val = qHDF_read_data(data, prefix + name, file_size, false);
            if (quadriga_lib::any_type_id(&val) > 0)
                c.par_names.push_back(name), c.par_data.push_back(std::move(val));
        }
    }

    // Channel name, number of snapshots, index of reference position
    c.name = qHDF_read_name(group);
    unsigned n_snapshots = 0;
    qHDF_read_attr(group, "NumSnap", &n_snapshots);
    qHDF_read_attr(group, "Initial_position", &c.initial_position, 1, false);

    // If there are no snapshots, we don't need to continue
    if (n_snapshots == 0)
        return c;

    // Number of transmitters and receivers: sizes of the coefficient datasets
    unsigned n_tx = 0, n_rx = 0;
    qHDF_read_attr(group, "NumTx", &n_tx);
    qHDF_read_attr(group, "NumRx", &n_rx);
    const unsigned long long n_snap = n_snapshots;

    // Center frequency in [Hz]: dataset of length [n_snap] or scalar attribute (NaN = not set, read back empty)
    if (qHDF_link_exists(group, "CenterFrequency"))
    {
        qHDF_dataset dset(group, "CenterFrequency", {n_snap}, file_size);
        c.center_frequency.set_size((arma::uword)n_snap);
        dset.read(c.center_frequency.memptr());
    }
    else
    {
        float value = NAN;
        if (qHDF_read_attr(group, "CenterFrequency", &value, 1, false) && !std::isnan(value))
            c.center_frequency = {(dtype)value};
    }

    // Positions and orientations: dataset [3, n_snap] or attribute [3, 1]; orientations are optional
    auto read_position = [&](const char *name, arma::Mat<dtype> &value, bool required)
    {
        if (qHDF_link_exists(group, name))
        {
            qHDF_dataset dset(group, name, {n_snap, 3}, file_size);
            value.set_size(3, (arma::uword)n_snap);
            dset.read(value.memptr());
        }
        else
        {
            float data[3];
            if (qHDF_read_attr(group, name, data, 3, required))
            {
                value.set_size(3, 1);
                value.at(0) = (dtype)data[0], value.at(1) = (dtype)data[1], value.at(2) = (dtype)data[2];
            }
        }
    };
    read_position("tx_position", c.tx_pos, true);
    read_position("rx_position", c.rx_pos, true);
    read_position("tx_orientation", c.tx_orientation, false);
    read_position("rx_orientation", c.rx_orientation, false);

    // Snapshot data: in the channel group for one snapshot, in groups "Snap_<i>" otherwise
    std::vector<unsigned> snaps = (n_snapshots == 1) ? std::vector<unsigned>{0} : qHDF_snapshot_groups(group, n_snapshots);
    if (n_snapshots != 1 && !snaps.empty() && n_snap - (unsigned long long)snaps.size() > qHDF_max_missing_snapshots)
        throw std::invalid_argument("Malformed HDF5 file: 'NumSnap' is much larger than the number of snapshots in the file.");

    // Snapshots without a group (files written before empty snapshots were stored) are padded with empty entries
    unsigned j_coeff = 0,
             j_delay = 0,
             j_path_gain = 0,
             j_path_length = 0,
             j_path_polarization = 0,
             j_path_angles = 0,
             j_path_fbs_pos = 0,
             j_path_lbs_pos = 0,
             j_no_interact = 0;

    const arma::uword n_rx_arma = (arma::uword)n_rx, n_tx_arma = (arma::uword)n_tx;

    for (unsigned i : snaps)
    {
        // Open group containing snapshot data
        qHDF_id snap_group;
        hid_t snap_id = group;
        if (n_snapshots != 1)
        {
            const std::string snap_name = "Snap_" + std::to_string(i);
            snap_group = qHDF_id(H5Gopen2(group, snap_name.c_str(), H5P_DEFAULT), H5Gclose);
            if (!snap_group.valid())
                throw std::invalid_argument("Malformed HDF5 file: group '" + snap_name + "' cannot be opened.");
            snap_id = snap_group;
        }

        // Number of paths (missing if the snapshot holds no path data)
        unsigned n_path = 0;
        qHDF_read_attr(snap_id, "NumPath", &n_path, 1, false);
        const unsigned long long np = n_path;
        const arma::uword n_path_arma = (arma::uword)n_path;

        // Channel coefficients, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
        if (qHDF_link_exists(snap_id, "coeff_re"))
        {
            qHDF_dataset dset_re(snap_id, "coeff_re", {np, n_tx, n_rx}, file_size), dset_im(snap_id, "coeff_im", {np, n_tx, n_rx}, file_size);
            for (unsigned j = j_coeff; j < i; ++j)
                c.coeff_re.push_back(arma::Cube<dtype>(n_rx_arma, n_tx_arma, 0ULL)),
                    c.coeff_im.push_back(arma::Cube<dtype>(n_rx_arma, n_tx_arma, 0ULL));
            j_coeff = i + 1;

            arma::Cube<dtype> val_re(n_rx_arma, n_tx_arma, n_path_arma, arma::fill::none);
            dset_re.read(val_re.memptr());
            c.coeff_re.push_back(std::move(val_re));

            arma::Cube<dtype> val_im(n_rx_arma, n_tx_arma, n_path_arma, arma::fill::none);
            dset_im.read(val_im.memptr());
            c.coeff_im.push_back(std::move(val_im));
        }

        // Path delays in seconds, vector (n_snap) of tensors of size [n_rx, n_tx, n_path] or [1, 1, n_path]
        if (qHDF_link_exists(snap_id, "delay"))
        {
            qHDF_dataset dset(snap_id, "delay", {np, qHDF_ANY, qHDF_ANY}, file_size);
            const arma::uword n_tx_local = (arma::uword)dset.dim(1), n_rx_local = (arma::uword)dset.dim(2);

            for (unsigned j = j_delay; j < i; ++j)
                c.delay.push_back(arma::Cube<dtype>(n_rx_local, n_tx_local, 0ULL));
            j_delay = i + 1;

            arma::Cube<dtype> val(n_rx_local, n_tx_local, n_path_arma, arma::fill::none);
            dset.read(val.memptr());
            c.delay.push_back(std::move(val));
        }

        // Path gain before antenna patterns, vector (n_snap) of vectors of length [n_path]
        if (qHDF_link_exists(snap_id, "path_gain"))
        {
            qHDF_dataset dset(snap_id, "path_gain", {np}, file_size);
            for (unsigned j = j_path_gain; j < i; ++j)
                c.path_gain.push_back(arma::Col<dtype>());
            j_path_gain = i + 1;

            arma::Col<dtype> val(n_path_arma, arma::fill::none);
            dset.read(val.memptr());
            c.path_gain.push_back(std::move(val));
        }

        // Absolute path length from TX to RX phase center, vector (n_snap) of vectors of length [n_path]
        if (qHDF_link_exists(snap_id, "path_length"))
        {
            qHDF_dataset dset(snap_id, "path_length", {np}, file_size);
            for (unsigned j = j_path_length; j < i; ++j)
                c.path_length.push_back(arma::Col<dtype>());
            j_path_length = i + 1;

            arma::Col<dtype> val(n_path_arma, arma::fill::none);
            dset.read(val.memptr());
            c.path_length.push_back(std::move(val));
        }

        // Polarization transfer function, vector (n_snap) of matrices of size [8, n_path], interleaved complex
        if (qHDF_link_exists(snap_id, "path_polarization"))
        {
            qHDF_dataset dset(snap_id, "path_polarization", {np, 8}, file_size);
            for (unsigned j = j_path_polarization; j < i; ++j)
                c.path_polarization.push_back(arma::Mat<dtype>(8ULL, 0ULL));
            j_path_polarization = i + 1;

            arma::Mat<dtype> val(8ULL, n_path_arma, arma::fill::none);
            dset.read(val.memptr());
            c.path_polarization.push_back(std::move(val));
        }

        // Departure and arrival angles, vector (n_snap) of matrices of size [n_path, 4], {AOD, EOD, AOA, EOA}
        if (qHDF_link_exists(snap_id, "path_angles"))
        {
            qHDF_dataset dset(snap_id, "path_angles", {4, np}, file_size);
            for (unsigned j = j_path_angles; j < i; ++j)
                c.path_angles.push_back(arma::Mat<dtype>(0, 4));
            j_path_angles = i + 1;

            arma::Mat<dtype> val(n_path_arma, 4ULL, arma::fill::none);
            dset.read(val.memptr());
            c.path_angles.push_back(std::move(val));
        }

        // First-bounce scatterer positions, matrices of size [3, n_path]
        if (qHDF_link_exists(snap_id, "path_fbs_pos"))
        {
            qHDF_dataset dset(snap_id, "path_fbs_pos", {np, 3}, file_size);
            for (unsigned j = j_path_fbs_pos; j < i; ++j)
                c.path_fbs_pos.push_back(arma::Mat<dtype>(3ULL, 0ULL));
            j_path_fbs_pos = i + 1;

            arma::Mat<dtype> val(3ULL, n_path_arma, arma::fill::none);
            dset.read(val.memptr());
            c.path_fbs_pos.push_back(std::move(val));
        }

        // Last-bounce scatterer positions, matrices of size [3, n_path]
        if (qHDF_link_exists(snap_id, "path_lbs_pos"))
        {
            qHDF_dataset dset(snap_id, "path_lbs_pos", {np, 3}, file_size);
            for (unsigned j = j_path_lbs_pos; j < i; ++j)
                c.path_lbs_pos.push_back(arma::Mat<dtype>(3, 0));
            j_path_lbs_pos = i + 1;

            arma::Mat<dtype> val(3ULL, n_path_arma, arma::fill::none);
            dset.read(val.memptr());
            c.path_lbs_pos.push_back(std::move(val));
        }

        // Number interaction points of a path with the environment, 0 = LOS, and their coordinates [3, sum]
        if (qHDF_link_exists(snap_id, "no_interact"))
        {
            qHDF_dataset dset(snap_id, "no_interact", {np}, file_size);
            for (unsigned j = j_no_interact; j < i; ++j)
                c.no_interact.push_back(arma::u32_vec()), c.interact_coord.push_back(arma::Mat<dtype>(3ULL, 0ULL));
            j_no_interact = i + 1;

            arma::u32_vec val1(n_path_arma, arma::fill::none);
            dset.read(val1.memptr());

            unsigned long long n_coord = 0;
            for (auto &v : val1)
                n_coord += (unsigned long long)v;

            arma::Mat<dtype> val;
            if (n_coord != 0)
            {
                qHDF_dataset dset_coord(snap_id, "interact_coord", {n_coord, 3}, file_size);
                val.set_size(3ULL, (arma::uword)n_coord);
                dset_coord.read(val.memptr());
            }
            else
                val.set_size(3ULL, 0ULL);

            c.no_interact.push_back(std::move(val1));
            c.interact_coord.push_back(std::move(val));
        }
    }

    // Fill missing data
    if (j_coeff != 0)
        for (unsigned j = j_coeff; j < n_snapshots; ++j)
            c.coeff_re.push_back(arma::Cube<dtype>(n_rx_arma, n_tx_arma, 0)), c.coeff_im.push_back(arma::Cube<dtype>(n_rx_arma, n_tx_arma, 0));
    if (j_delay != 0)
        for (unsigned j = j_delay; j < n_snapshots; ++j)
            c.delay.push_back(arma::Cube<dtype>(n_rx_arma, n_tx_arma, 0));
    if (j_path_gain != 0)
        for (unsigned j = j_path_gain; j < n_snapshots; ++j)
            c.path_gain.push_back(arma::Col<dtype>());
    if (j_path_length != 0)
        for (unsigned j = j_path_length; j < n_snapshots; ++j)
            c.path_length.push_back(arma::Col<dtype>());
    if (j_path_polarization != 0)
        for (unsigned j = j_path_polarization; j < n_snapshots; ++j)
            c.path_polarization.push_back(arma::Mat<dtype>(8, 0));
    if (j_path_angles != 0)
        for (unsigned j = j_path_angles; j < n_snapshots; ++j)
            c.path_angles.push_back(arma::Mat<dtype>(0, 4));
    if (j_path_fbs_pos != 0)
        for (unsigned j = j_path_fbs_pos; j < n_snapshots; ++j)
            c.path_fbs_pos.push_back(arma::Mat<dtype>(3, 0));
    if (j_path_lbs_pos != 0)
        for (unsigned j = j_path_lbs_pos; j < n_snapshots; ++j)
            c.path_lbs_pos.push_back(arma::Mat<dtype>(3, 0));
    if (j_no_interact != 0)
        for (unsigned j = j_no_interact; j < n_snapshots; ++j)
            c.no_interact.push_back(arma::u32_vec()), c.interact_coord.push_back(arma::Mat<dtype>(3ULL, 0ULL));

    return c;
}

template quadriga_lib::channel<float> quadriga_lib::hdf5_read_channel(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw);

template quadriga_lib::channel<double> quadriga_lib::hdf5_read_channel(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw);

/*!MD
# hdf5_reshape_layout
Reshape the 4D storage layout of an existing HDF5 channel file

- Updates `{nx, ny, nz, nw}` of an existing file; total slot count `nx × ny × nz × nw` must remain unchanged.
- Throws if the new layout violates the total-count constraint or has more than 4294967295 (2^32 - 1) slots.

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
    if (!qHDF_file_exists(fn))
        throw std::invalid_argument("File does not exist.");

    // Checked in 64 bits, before the file is opened
    const unsigned long long n_data_new = qHDF_slot_count(nx, ny, nz, nw);

    qHDF_error_guard error_guard;

    // Open file for writing
    qHDF_id file = qHDF_open_file(fn, H5F_ACC_RDWR);
    qHDF_index idx = qHDF_read_index(file, qHDF_file_size(file), false);

    unsigned long long n_data_old = 0;
    try
    {
        n_data_old = qHDF_slot_count(idx.dims[0], idx.dims[1], idx.dims[2], idx.dims[3]);
    }
    catch (const std::invalid_argument &)
    {
        throw std::invalid_argument("Storage index in HDF file is corrupted.");
    }

    if (n_data_new != n_data_old)
        throw std::invalid_argument("Mismatch in number of elements in storage index.");

    unsigned ChannelDims[4] = {nx, ny, nz, nw};
    qHDF_id dset = qHDF_check(qHDF_id(H5Dopen2(file, "ChannelDims", H5P_DEFAULT), H5Dclose), "cannot open 'ChannelDims'");
    qHDF_check(H5Dwrite(dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, ChannelDims), "cannot write 'ChannelDims'");
    dset.reset();

    qHDF_check(file.close(), "cannot close file", fn.c_str());
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
- Element types: float, double, 32- and 64-bit signed and unsigned integers. Datasets of other types or with more
  than 3 dimensions are returned as empty `std::any`.
- Strings: fixed-length strings are returned without trailing NUL characters (an empty string written by
  [[hdf5_write_dset]] comes back empty); variable-length strings (the default of h5py and MATLAB) are supported.

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
    if (!qHDF_file_exists(fn))
        throw std::invalid_argument("File does not exist.");

    qHDF_error_guard error_guard;

    // Open file for reading
    qHDF_id file = qHDF_open_file(fn, H5F_ACC_RDONLY);

    // Get channel ID
    const unsigned long long file_size = qHDF_file_size(file);
    const unsigned channel_index = qHDF_channel_id(file, file_size, ix, iy, iz, iw);
    if (channel_index == 0)
        return std::any();

    // Read data
    qHDF_id group = qHDF_open_channel_group(file, channel_index);
    if (!qHDF_link_exists(group, "Data"))
        return std::any();
    qHDF_id data = qHDF_open_data_group(group);
    return qHDF_read_data(data, prefix + par_name, file_size);
}

/*!MD
# hdf5_read_dset_names
Read names of unstructured datasets stored at a 4D slot in an HDF5 file

- Finds all datasets whose HDF5 name starts with `prefix` at slot `(ix, iy, iz, iw)`; returned names exclude the prefix.

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
    if (!qHDF_file_exists(fn))
        throw std::invalid_argument("File does not exist.");

    qHDF_error_guard error_guard;

    // Open file for reading
    qHDF_id file = qHDF_open_file(fn, H5F_ACC_RDONLY);

    // Get channel ID
    const unsigned long long file_size = qHDF_file_size(file);
    const unsigned channel_index = qHDF_channel_id(file, file_size, ix, iy, iz, iw);
    if (channel_index == 0)
        return 0ULL;

    // Read names
    qHDF_id group = qHDF_open_channel_group(file, channel_index);
    if (!qHDF_link_exists(group, "Data"))
        return 0ULL;
    qHDF_id data = qHDF_open_data_group(group);
    return (arma::uword)qHDF_read_par_names(data, par_names, prefix);
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

    qHDF_error_guard error_guard;

    // Create file
    if (!qHDF_file_exists(fn))
        quadriga_lib::hdf5_create(fn);

    // Open file for writing
    qHDF_id file = qHDF_open_file(fn, H5F_ACC_RDWR);

    // Slot and its channel ID; an empty slot gets a new group
    qHDF_index idx = qHDF_read_index(file, qHDF_file_size(file));
    const long long slot = qHDF_slot(idx, ix, iy, iz, iw);
    if (slot < 0)
        throw std::invalid_argument("Index out of bound.");
    const bool new_slot = idx.order[(size_t)slot] == 0;
    const unsigned channel_index = new_slot ? qHDF_new_channel_id(file, idx.order) : idx.order[(size_t)slot];

    qHDF_unlink_guard remove_on_error(file, qHDF_group_name(channel_index), new_slot);
    {
        qHDF_id group = new_slot ? qHDF_create_channel_group(file, channel_index) : qHDF_open_channel_group(file, channel_index);
        qHDF_id data = qHDF_link_exists(group, "Data") ? qHDF_open_data_group(group) : qHDF_create_data_group(group);
        qHDF_write_par(data, name, par_data);
    }
    if (new_slot)
    {
        idx.order[(size_t)slot] = channel_index;
        qHDF_write_order(file, idx.order);
        remove_on_error.release();
    }

    qHDF_check(file.close(), "cannot close file", fn.c_str());
}
