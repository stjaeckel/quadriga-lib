// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Files written by the library, opened with the HDF5 C library: the complete object tree (groups, datasets and
// attributes with their names, element types, dataspaces and values) must be the one that the written channel
// objects define. This shows that HDF5 readers other than the library (h5py, MATLAB, HDFView) see the documented
// tree. The behavior of the library functions themselves is tested in test_hdf_functions.cpp.
// This file is compiled only if CMake finds an HDF5 installation (version 1.12 or newer); see CMakeLists.txt.
//
// Oracle: the tree derived from the channel objects by the storage rules of the channel file format.
//   /                    datasets ChannelDims (uint32 [4]), Order (uint32 [n_slot], channel ID per slot, 0 = empty)
//                        and Version (uint32 scalar, 2 or 3); version 3 adds ChannelRef (one object reference per
//                        slot: the group of the slot, a null reference for an empty slot)
//   /channel_<id>        attributes Name (fixed-length string of 256 bytes, NUL-terminated and NUL-padded), NumSnap,
//                        NumTx, NumRx (uint32), Initial_position (int32), CenterFrequency (float), tx_position,
//                        rx_position (float [3], NaN if not set), tx_orientation, rx_orientation (float [3], zeros
//                        if not set). A value per snapshot is a float dataset of the same name; the attribute keeps
//                        its default then. A channel without structured data keeps all defaults (and no name).
//   snapshot data        in the channel group (one snapshot) or in groups Snap_<i>: attribute NumPath (uint32) and
//                        float datasets coeff_re, coeff_im, delay, path_gain, ... and uint32 no_interact
//   /channel_<id>/Data   unstructured datasets par_<name> in creation order: the element type of the written object,
//                        a scalar dataspace for scalars, rank 1 for vectors (arma::Row too), rank 2 for matrices,
//                        rank 3 for cubes; strings as fixed-length strings (an empty string as one NUL byte)
// Extents are given in HDF5 order (slowest dimension first): an Armadillo object [rows, cols, slices] has the
// extents {slices, cols, rows} and the same memory order.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <hdf5.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>

namespace
{
    // Expected dataset or attribute: file type, extents (empty = scalar dataspace) and the stored bytes
    struct value
    {
        hid_t type = H5I_INVALID_HID; // Predefined type, e.g. H5T_IEEE_F32LE; H5T_C_S1 for fixed-length strings
        std::vector<hsize_t> dims;
        std::vector<unsigned char> bytes; // Strings: all bytes including the NUL padding
    };

    // The expected file: every object path (groups and datasets) with its attributes, the data of the datasets,
    // and the link names of the groups that keep the creation order
    struct tree
    {
        std::map<std::string, std::map<std::string, value>> attributes;
        std::map<std::string, value> datasets;
        std::map<std::string, std::vector<std::string>> creation_order;

        void add_dataset(const std::string &path, value v)
        {
            attributes[path];
            datasets[path] = std::move(v);
        }
    };

    template <typename T>
    value raw(hid_t type, std::vector<hsize_t> dims, const T *data, size_t n)
    {
        return {type, std::move(dims), std::vector<unsigned char>((const unsigned char *)data, (const unsigned char *)(data + n))};
    }

    // Any Armadillo object, stored in single precision
    template <typename A>
    value as_float(const A &x, std::vector<hsize_t> dims)
    {
        std::vector<float> f(x.n_elem);
        for (arma::uword i = 0; i < x.n_elem; ++i)
            f[i] = (float)x[i];
        return raw(H5T_IEEE_F32LE, std::move(dims), f.data(), f.size());
    }

    // Fixed-length string of 'size' bytes, NUL-padded
    value fixed_string(const std::string &s, size_t size)
    {
        value v{H5T_C_S1, {}, std::vector<unsigned char>(size, 0)};
        std::memcpy(v.bytes.data(), s.data(), std::min(s.size(), size));
        return v;
    }

    // Unstructured data: element type and rank from the any_type_id code
    value par_value(const std::any &x)
    {
        unsigned long long d[3];
        void *ptr = nullptr;
        const int id = quadriga_lib::any_type_id(&x, d, &ptr);
        if (id == 9)
            return fixed_string(std::any_cast<std::string>(x), std::max((size_t)d[0], (size_t)1));
        const hid_t types[6] = {H5T_IEEE_F32LE, H5T_IEEE_F64LE, H5T_STD_U64LE, H5T_STD_I64LE, H5T_STD_U32LE, H5T_STD_I32LE};
        std::vector<hsize_t> dims;
        if (id >= 20 && id < 30)
            dims = {d[1], d[0]};
        else if (id >= 30 && id < 40)
            dims = {d[2], d[1], d[0]};
        else if (id >= 40)
            dims = {d[0]};
        const size_t n = (size_t)(d[0] * d[1] * d[2]), elem_size = H5Tget_size(types[id % 10]);
        return raw(types[id % 10], dims, (const unsigned char *)ptr, n * elem_size);
    }

    // Group /channel_<id> of a channel written with hdf5_write (or created by hdf5_write_dset: no structured data)
    template <typename dtype>
    void add_channel(tree &t, unsigned id, const quadriga_lib::channel<dtype> &c)
    {
        const std::string g = "/channel_" + std::to_string(id);
        const bool structured = !c.empty();
        const unsigned n_snap = structured ? (unsigned)c.n_snap() : 0;
        const unsigned n_tx = structured ? (unsigned)c.n_tx() : 0, n_rx = structured ? (unsigned)c.n_rx() : 0;
        const int initial_position = structured ? c.initial_position : 0;
        const float nan3[3] = {NAN, NAN, NAN}, zero3[3] = {0.0f, 0.0f, 0.0f};

        auto &a = t.attributes[g];
        a["Name"] = fixed_string(structured ? c.name : "", 256);
        a["NumSnap"] = raw(H5T_STD_U32LE, {}, &n_snap, 1);
        a["NumTx"] = raw(H5T_STD_U32LE, {}, &n_tx, 1);
        a["NumRx"] = raw(H5T_STD_U32LE, {}, &n_rx, 1);
        a["Initial_position"] = raw(H5T_STD_I32LE, {}, &initial_position, 1);

        // One value: attribute; one value per snapshot: dataset (the attribute keeps its default)
        a["CenterFrequency"] = c.center_frequency.n_elem == 1 ? as_float(c.center_frequency, {}) : raw(H5T_IEEE_F32LE, {}, nan3, 1);
        if (c.center_frequency.n_elem > 1)
            t.add_dataset(g + "/CenterFrequency", as_float(c.center_frequency, {n_snap}));
        auto position = [&](const arma::Mat<dtype> &x, const char *name, const float *default_value)
        {
            a[name] = x.n_cols == 1 ? as_float(x, {3}) : raw(H5T_IEEE_F32LE, {3}, default_value, 3);
            if (x.n_cols > 1)
                t.add_dataset(g + "/" + name, as_float(x, {x.n_cols, 3}));
        };
        position(c.tx_pos, "tx_position", nan3);
        position(c.rx_pos, "rx_position", nan3);
        position(c.tx_orientation, "tx_orientation", zero3);
        position(c.rx_orientation, "rx_orientation", zero3);

        // Snapshot data, if the channel has any per-snapshot field
        const size_t ns = n_snap;
        const bool has_snapshot_data = c.coeff_re.size() == ns || c.delay.size() == ns || c.path_gain.size() == ns ||
                                       c.path_length.size() == ns || c.path_polarization.size() == ns ||
                                       c.path_angles.size() == ns || c.path_fbs_pos.size() == ns ||
                                       c.path_lbs_pos.size() == ns || c.no_interact.size() == ns;
        const arma::uvec n_path = c.n_path();
        for (size_t s = 0; s < ns && has_snapshot_data; ++s)
        {
            const std::string sg = ns == 1 ? g : g + "/Snap_" + std::to_string(s);
            const unsigned np = (unsigned)n_path(s);
            t.attributes[sg]["NumPath"] = raw(H5T_STD_U32LE, {}, &np, 1);

            auto cube = [&](const std::vector<arma::Cube<dtype>> &x, const char *name)
            {
                if (x.size() == ns)
                    t.add_dataset(sg + "/" + name, as_float(x[s], {x[s].n_slices, x[s].n_cols, x[s].n_rows}));
            };
            auto mat = [&](const std::vector<arma::Mat<dtype>> &x, const char *name)
            {
                if (x.size() == ns)
                    t.add_dataset(sg + "/" + name, as_float(x[s], {x[s].n_cols, x[s].n_rows}));
            };
            auto col = [&](const std::vector<arma::Col<dtype>> &x, const char *name)
            {
                if (x.size() == ns)
                    t.add_dataset(sg + "/" + name, as_float(x[s], {x[s].n_elem}));
            };
            cube(c.coeff_re, "coeff_re");
            cube(c.coeff_im, "coeff_im");
            cube(c.delay, "delay");
            col(c.path_gain, "path_gain");
            col(c.path_length, "path_length");
            mat(c.path_polarization, "path_polarization");
            mat(c.path_angles, "path_angles");
            mat(c.path_fbs_pos, "path_fbs_pos");
            mat(c.path_lbs_pos, "path_lbs_pos");
            if (c.no_interact.size() == ns)
                t.add_dataset(sg + "/no_interact", raw(H5T_STD_U32LE, {np}, c.no_interact[s].memptr(), np));
            mat(c.interact_coord, "interact_coord");
        }

        if (!c.par_names.empty())
        {
            t.attributes[g + "/Data"];
            for (size_t i = 0; i < c.par_names.size(); ++i)
            {
                t.add_dataset(g + "/Data/par_" + c.par_names[i], par_value(c.par_data[i]));
                t.creation_order[g + "/Data"].push_back("par_" + c.par_names[i]);
            }
        }
    }

    // Root datasets of a file with the given layout and channel IDs
    tree root_tree(const std::vector<unsigned> &layout, const std::vector<unsigned> &order, unsigned version)
    {
        tree t;
        t.attributes["/"];
        t.add_dataset("/ChannelDims", raw(H5T_STD_U32LE, {4}, layout.data(), 4));
        t.add_dataset("/Order", raw(H5T_STD_U32LE, {order.size()}, order.data(), order.size()));
        t.add_dataset("/Version", raw(H5T_STD_U32LE, {}, &version, 1));
        if (version == 3) // Data checked by dereferencing
            t.add_dataset("/ChannelRef", {H5T_STD_REF_OBJ, {order.size()}, {}});
        return t;
    }

    // Owns an HDF5 identifier
    struct id_guard
    {
        hid_t id;
        herr_t (*close)(hid_t);
        ~id_guard()
        {
            if (id >= 0)
                close(id);
        }
        operator hid_t() const { return id; }
    };

    // Element type, dataspace and data of a dataset or attribute; 'read' reads all elements as the given memory type
    template <typename Read>
    void check_value(hid_t type, hid_t space, const value &v, Read read)
    {
        if (v.type == H5T_C_S1)
        {
            CHECK(H5Tget_class(type) == H5T_STRING);
            CHECK(H5Tis_variable_str(type) == 0);
            CHECK(H5Tget_size(type) == v.bytes.size());
            CHECK(H5Tget_strpad(type) == H5T_STR_NULLTERM);
            CHECK(H5Tget_cset(type) == H5T_CSET_ASCII);
        }
        else
            CHECK(H5Tequal(type, v.type) > 0);

        if (v.dims.empty())
            CHECK(H5Sget_simple_extent_type(space) == H5S_SCALAR);
        else
        {
            CHECK(H5Sget_simple_extent_type(space) == H5S_SIMPLE);
            REQUIRE(H5Sget_simple_extent_ndims(space) == (int)v.dims.size());
            std::vector<hsize_t> dims(v.dims.size()), max_dims(v.dims.size());
            H5Sget_simple_extent_dims(space, dims.data(), max_dims.data());
            CHECK(dims == v.dims);
            CHECK(max_dims == v.dims);
        }

        if (v.type == H5T_STD_REF_OBJ || v.bytes.empty())
            return;
        std::vector<unsigned char> data(v.bytes.size());
        REQUIRE(read(v.type == H5T_C_S1 ? type : v.type, data.data()) >= 0);
        if (H5Tequal(v.type, H5T_IEEE_F32LE) > 0) // NaN equals NaN
            for (size_t i = 0; i < data.size(); i += 4)
            {
                float x, y;
                std::memcpy(&x, &data[i], 4), std::memcpy(&y, &v.bytes[i], 4);
                CHECK((x == y || (std::isnan(x) && std::isnan(y))));
            }
        else
            CHECK(data == v.bytes);
    }

    // Compares the file with the expected tree, object by object
    void check_file(const std::string &fn, const tree &expected)
    {
        id_guard file{H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
        REQUIRE(file >= 0);

        // All objects reachable from the root group, and no other
        std::map<std::string, H5O_type_t> objects;
        auto collect_object = [](hid_t, const char *name, const H5O_info2_t *info, void *objects) -> herr_t
        {
            const std::string path = std::strcmp(name, ".") == 0 ? "/" : "/" + std::string(name);
            (*(std::map<std::string, H5O_type_t> *)objects)[path] = info->type;
            return 0;
        };
        REQUIRE(H5Ovisit3(file, H5_INDEX_NAME, H5_ITER_INC, collect_object, &objects, H5O_INFO_BASIC) >= 0);
        std::vector<std::string> missing, unexpected;
        for (auto &e : expected.attributes)
            if (objects.count(e.first) == 0)
                missing.push_back(e.first);
        for (auto &o : objects)
            if (expected.attributes.count(o.first) == 0)
                unexpected.push_back(o.first);
        CHECK(missing == std::vector<std::string>());
        CHECK(unexpected == std::vector<std::string>());

        for (auto &e : expected.attributes)
        {
            if (objects.count(e.first) == 0)
                continue;
            INFO("object " << e.first);
            const bool is_dataset = expected.datasets.count(e.first) != 0;
            CHECK(objects[e.first] == (is_dataset ? H5O_TYPE_DATASET : H5O_TYPE_GROUP));

            // Attributes: the expected names and no other, with their values
            id_guard object{H5Oopen(file, e.first.c_str(), H5P_DEFAULT), H5Oclose};
            REQUIRE(object >= 0);
            std::vector<std::string> names;
            hsize_t position = 0;
            auto collect_name = [](hid_t, const char *name, const H5A_info_t *, void *names) -> herr_t
            {
                ((std::vector<std::string> *)names)->push_back(name);
                return 0;
            };
            H5Aiterate2(object, H5_INDEX_NAME, H5_ITER_INC, &position, collect_name, &names);
            std::vector<std::string> expected_names;
            for (auto &a : e.second)
                expected_names.push_back(a.first); // Sorted by name, as listed by H5Aiterate2
            CHECK(names == expected_names);
            for (auto &a : e.second)
            {
                INFO("attribute " << a.first);
                id_guard attr{H5Aopen(object, a.first.c_str(), H5P_DEFAULT), H5Aclose};
                REQUIRE(attr >= 0);
                id_guard type{H5Aget_type(attr), H5Tclose}, space{H5Aget_space(attr), H5Sclose};
                check_value(type, space, a.second, [&](hid_t mem_type, void *buf)
                            { return H5Aread(attr, mem_type, buf); });
            }

            if (is_dataset)
            {
                const value &v = expected.datasets.at(e.first);
                id_guard type{H5Dget_type(object), H5Tclose}, space{H5Dget_space(object), H5Sclose};
                check_value(type, space, v, [&](hid_t mem_type, void *buf)
                            { return H5Dread(object, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf); });
            }
        }

        // Groups that keep the creation order list their links in write order
        for (auto &g : expected.creation_order)
        {
            INFO("group " << g.first);
            id_guard group{H5Gopen2(file, g.first.c_str(), H5P_DEFAULT), H5Gclose};
            std::vector<std::string> names;
            hsize_t position = 0;
            auto collect_name = [](hid_t, const char *name, const H5L_info2_t *, void *names) -> herr_t
            {
                ((std::vector<std::string> *)names)->push_back(name);
                return 0;
            };
            CHECK(H5Literate2(group, H5_INDEX_CRT_ORDER, H5_ITER_INC, &position, collect_name, &names) >= 0);
            CHECK(names == g.second);
        }
    }

    // Version 3 only: ChannelRef holds the group of each slot, a null reference for an empty slot
    void check_channel_ref(const std::string &fn, const std::vector<unsigned> &order)
    {
        id_guard file{H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
        id_guard dset{H5Dopen2(file, "ChannelRef", H5P_DEFAULT), H5Dclose};
        REQUIRE(dset >= 0);
        std::vector<hobj_ref_t> refs(order.size());
        REQUIRE(H5Dread(dset, H5T_STD_REF_OBJ, H5S_ALL, H5S_ALL, H5P_DEFAULT, refs.data()) >= 0);
        for (size_t slot = 0; slot < order.size(); ++slot)
        {
            INFO("slot " << slot);
            if (order[slot] == 0)
            {
                CHECK(refs[slot] == 0);
                continue;
            }
            id_guard target{H5Rdereference2(dset, H5P_DEFAULT, H5R_OBJECT, &refs[slot]), H5Oclose};
            REQUIRE(target >= 0);
            H5O_info2_t target_info, group_info;
            H5Oget_info3(target, &target_info, H5O_INFO_BASIC);
            const std::string group = "/channel_" + std::to_string(order[slot]);
            REQUIRE(H5Oget_info_by_name3(file, group.c_str(), &group_info, H5O_INFO_BASIC, H5P_DEFAULT) >= 0);
            int cmp = 1;
            H5Otoken_cmp(file, &target_info.token, &group_info.token, &cmp);
            CHECK(cmp == 0);
        }
    }

    // The file version (2 or 3) and, for version 3, the channel references; then the whole tree
    template <typename Build>
    void check_file_with_version(const std::string &fn, const std::vector<unsigned> &layout,
                                 const std::vector<unsigned> &order, Build add_channels)
    {
        unsigned version = 0;
        {
            id_guard file{H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
            id_guard dset{H5Dopen2(file, "Version", H5P_DEFAULT), H5Dclose};
            REQUIRE(dset >= 0);
            REQUIRE(H5Dread(dset, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &version) >= 0);
        }
        CHECK((version == 2 || version == 3));
        tree t = root_tree(layout, order, version);
        add_channels(t);
        check_file(fn, t);
        if (version == 3)
            check_channel_ref(fn, order);
    }

    template <typename A>
    A filled(A x, double start, double scale = 1.0)
    {
        x.imbue([&]()
                { return (typename A::elem_type)((start += 0.37) * scale); });
        return x;
    }

    // A channel with every structured field, n_rx = 2, n_tx = 3; odd snapshots have delays of size [1, 1, n_path]
    template <typename dtype>
    quadriga_lib::channel<dtype> full_channel(const std::string &name, const std::vector<arma::uword> &n_path, bool per_snapshot)
    {
        using cube = arma::Cube<dtype>;
        using mat = arma::Mat<dtype>;
        using col = arma::Col<dtype>;
        const arma::uword n_snap = n_path.size(), n_pos = per_snapshot ? n_snap : 1;
        quadriga_lib::channel<dtype> c;
        c.name = name;
        c.initial_position = 1;
        c.center_frequency = filled(col(n_pos), 1.0, 1e9);
        c.tx_pos = filled(mat(3, n_pos), 1.0);
        c.rx_pos = filled(mat(3, n_pos), 10.0);
        if (per_snapshot)
        {
            c.tx_orientation = filled(mat(3, n_pos), -1.0);
            c.rx_orientation = filled(mat(3, 1), -2.0);
        }
        for (arma::uword s = 0; s < n_snap; ++s)
        {
            const arma::uword np = n_path[s];
            const double o = 100.0 * (double)s;
            c.coeff_re.push_back(filled(cube(2, 3, np), o));
            c.coeff_im.push_back(filled(cube(2, 3, np), o + 10.0));
            c.delay.push_back(filled(s % 2 ? cube(1, 1, np) : cube(2, 3, np), o + 20.0, 1e-9));
            c.path_gain.push_back(filled(col(np), o + 30.0));
            c.path_length.push_back(filled(col(np), o + 40.0));
            c.path_polarization.push_back(filled(mat(8, np), o + 50.0));
            c.path_angles.push_back(filled(mat(np, 4), o + 60.0));
            c.path_fbs_pos.push_back(filled(mat(3, np), o + 70.0));
            c.path_lbs_pos.push_back(filled(mat(3, np), o + 80.0));
            arma::u32_vec no_interact(np);
            for (arma::uword p = 0; p < np; ++p)
                no_interact(p) = (unsigned)(p % 3);
            c.no_interact.push_back(no_interact);
            c.interact_coord.push_back(filled(mat(3, arma::accu(no_interact)), o + 90.0));
        }
        return c;
    }
}

TEST_CASE("hdf5_write - the HDF5 C library reads the documented object tree")
{
    const std::string fn = "test_hdf5_interop.hdf5";
    std::remove(fn.c_str());

    // A new file: the root datasets only
    quadriga_lib::hdf5_create(fn, 3, 2);
    check_file_with_version(fn, {3, 2, 1, 1}, {0, 0, 0, 0, 0, 0}, [](tree &) {});

    // Slot 0: three snapshots (one without paths), values per snapshot, unstructured data of several types and ranks
    auto c0 = full_channel<double>(std::string(255, 'N'), {3, 0, 2}, true);
    c0.par_names = {"string", "empty_string", "scalar_float", "scalar_ullong", "col_int", "row_double", "mat_uword",
                    "mat_float_0x3", "cube_sword", "cube_double"};
    c0.par_data = {std::string("abc"), std::string(), 3.14f, 12341100000ULL, arma::Col<int>{-1, 2, -3},
                   arma::rowvec{0.5, 1.5}, arma::umat{{1, 2}, {3, 4}, {5, 6}}, arma::fmat(0, 3),
                   arma::icube(2, 3, 2, arma::fill::value(-7)), filled(arma::cube(3, 1, 2), 5.0)};
    quadriga_lib::hdf5_write(&c0, fn, 0);

    // Slot 1: one snapshot written as float, an empty name, positions for all snapshots, no orientations
    auto c1 = full_channel<float>("", {4}, false);
    quadriga_lib::hdf5_write(&c1, fn, 1);

    // Slot 2: an empty slot that gets unstructured data with hdf5_write_dset
    quadriga_lib::channel<double> c2;
    c2.par_names = {"note", "values"};
    c2.par_data = {std::string("note"), arma::Mat<unsigned>(2, 2, arma::fill::value(3u))};
    for (size_t i = 0; i < c2.par_names.size(); ++i)
        quadriga_lib::hdf5_write_dset(fn, c2.par_names[i], &c2.par_data[i], 2);

    // Slot 3: written, then overwritten; the old group is removed
    auto old = full_channel<double>("old", {2, 2}, false);
    old.par_names = {"old"};
    old.par_data = {1.0};
    quadriga_lib::hdf5_write(&old, fn, 0, 1);
    auto c3 = full_channel<double>("new", {1, 2}, false);
    c3.par_names = {"new"};
    c3.par_data = {2};
    quadriga_lib::hdf5_write(&c3, fn, 0, 1);

    // Slot 4: a channel without unstructured data, extended with hdf5_write_dset; then the layout is reshaped
    auto c4 = full_channel<double>(std::string(20, 'M'), {2}, true);
    quadriga_lib::hdf5_write(&c4, fn, 1, 1);
    c4.par_names = {"added"};
    c4.par_data = {arma::vec{1.0, 2.0}};
    quadriga_lib::hdf5_write_dset(fn, "added", &c4.par_data[0], 1, 1);
    quadriga_lib::hdf5_reshape_layout(fn, 2, 3);

    // Channel IDs: one above the largest ID in the file
    check_file_with_version(fn, {2, 3, 1, 1}, {1, 2, 3, 5, 6, 0}, [&](tree &t)
                            {
                                add_channel(t, 1, c0);
                                add_channel(t, 2, c1);
                                add_channel(t, 3, c2);
                                add_channel(t, 5, c3);
                                add_channel(t, 6, c4); });

    std::remove(fn.c_str());
}
