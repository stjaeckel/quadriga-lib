// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Files written by the library, opened with the HDF5 C library: the complete object tree (groups, datasets and
// attributes with their names, element types, dataspaces and values) must be the one that the written channel
// objects define. This shows that HDF5 readers other than the library (h5py, MATLAB, HDFView) see the documented
// tree. The behavior of the library functions themselves is tested in test_hdf_functions.cpp.
// Further cases: the attribute count that the HDF5 C library reports for every object; the intermediate states of an
// interrupted overwrite as the HDF5 C library sees them (including a read-write session, which must not change the
// file); 8000 root links in the root link directory; and files modified with the HDF5 C library, which the library
// rejects or reads consistently with the modified tree.
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
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_lib.hpp"

#include <hdf5.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
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

namespace
{
    const std::string unsupported_message =
        "Unsupported HDF5 file: quadriga-lib reads only channel files of version 3 that it wrote itself; files written "
        "by quadriga-lib 0.12.4 and earlier (version 2) and files modified with other HDF5 tools are not supported. "
        "Version-2 files can be converted with tools/hdf5_convert_v2.py.";

    std::vector<unsigned char> load(const std::string &fn)
    {
        std::ifstream f(fn, std::ios::binary);
        return std::vector<unsigned char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    void save(const std::string &fn, const std::vector<unsigned char> &b)
    {
        std::ofstream(fn, std::ios::binary | std::ios::trunc).write((const char *)b.data(), (std::streamsize)b.size());
    }

    // End-of-file address of the superblock (version 2: bytes 28-35)
    size_t eof_address(const std::vector<unsigned char> &b)
    {
        size_t eof = 0;
        for (int i = 7; i >= 0; --i)
            eof = (eof << 8) | b[28 + i];
        return eof;
    }

    // 1 snapshot, SISO, 1 path; the coefficient holds 'value'
    quadriga_lib::channel<double> minimal_channel(double value)
    {
        quadriga_lib::channel<double> c;
        c.tx_pos = arma::vec({0.0, 0.0, value});
        c.rx_pos = arma::vec({10.0, 0.0, 1.5});
        c.coeff_re = {arma::cube(1, 1, 1, arma::fill::value(value))};
        c.coeff_im = {arma::cube(1, 1, 1, arma::fill::zeros)};
        c.delay = {arma::cube(1, 1, 1, arma::fill::value(1e-6))};
        return c;
    }

    // The file of the case "the HDF5 C library reads the documented object tree" (layout 2 x 3, IDs {1, 2, 3, 5, 6,
    // 0}); 'step' is called after every change
    void reference_file(const std::string &fn, const std::function<void()> &step = []() {})
    {
        std::remove(fn.c_str());
        quadriga_lib::hdf5_create(fn, 3, 2);
        step();
        auto c0 = full_channel<double>(std::string(255, 'N'), {3, 0, 2}, true);
        c0.par_names = {"string", "empty_string", "scalar_float", "scalar_ullong", "col_int", "row_double", "mat_uword",
                        "mat_float_0x3", "cube_sword", "cube_double"};
        c0.par_data = {std::string("abc"), std::string(), 3.14f, 12341100000ULL, arma::Col<int>{-1, 2, -3},
                       arma::rowvec{0.5, 1.5}, arma::umat{{1, 2}, {3, 4}, {5, 6}}, arma::fmat(0, 3),
                       arma::icube(2, 3, 2, arma::fill::value(-7)), filled(arma::cube(3, 1, 2), 5.0)};
        quadriga_lib::hdf5_write(&c0, fn, 0);
        step();
        auto c1 = full_channel<float>("", {4}, false);
        quadriga_lib::hdf5_write(&c1, fn, 1);
        step();
        std::any note = std::string("note"), values = arma::Mat<unsigned>(2, 2, arma::fill::value(3u));
        quadriga_lib::hdf5_write_dset(fn, "note", &note, 2);
        step();
        quadriga_lib::hdf5_write_dset(fn, "values", &values, 2);
        step();
        auto old = full_channel<double>("old", {2, 2}, false);
        old.par_names = {"old"};
        old.par_data = {1.0};
        quadriga_lib::hdf5_write(&old, fn, 0, 1);
        step();
        auto c3 = full_channel<double>("new", {1, 2}, false);
        c3.par_names = {"new"};
        c3.par_data = {2};
        quadriga_lib::hdf5_write(&c3, fn, 0, 1);
        step();
        auto c4 = full_channel<double>(std::string(20, 'M'), {2}, true);
        quadriga_lib::hdf5_write(&c4, fn, 1, 1);
        step();
        std::any added = arma::vec{1.0, 2.0};
        quadriga_lib::hdf5_write_dset(fn, "added", &added, 1, 1);
        step();
        quadriga_lib::hdf5_reshape_layout(fn, 2, 3);
        step();
    }

    // Number of objects; objects whose attribute count (H5Oget_info3) differs from the number of attributes that
    // H5Aiterate2 lists are added to 'mismatches'
    size_t check_attribute_counts(const std::string &fn, std::vector<std::string> &mismatches)
    {
        id_guard file{H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
        REQUIRE(file >= 0);
        std::vector<std::string> names;
        auto collect = [](hid_t, const char *name, const H5O_info2_t *, void *names) -> herr_t
        {
            ((std::vector<std::string> *)names)->push_back(name);
            return 0;
        };
        REQUIRE(H5Ovisit3(file, H5_INDEX_NAME, H5_ITER_INC, collect, &names, H5O_INFO_BASIC) >= 0);
        for (const std::string &name : names)
        {
            id_guard object{H5Oopen(file, name.c_str(), H5P_DEFAULT), H5Oclose};
            REQUIRE(object >= 0);
            H5O_info2_t info;
            REQUIRE(H5Oget_info3(object, &info, H5O_INFO_NUM_ATTRS) >= 0);
            hsize_t position = 0, n_listed = 0;
            auto count = [](hid_t, const char *, const H5A_info_t *, void *n) -> herr_t
            {
                ++*(hsize_t *)n;
                return 0;
            };
            REQUIRE(H5Aiterate2(object, H5_INDEX_NAME, H5_ITER_INC, &position, count, &n_listed) >= 0);
            if (info.num_attrs != n_listed)
                mismatches.push_back(name + ": " + std::to_string(info.num_attrs) + " != " + std::to_string(n_listed));
        }
        return names.size();
    }

    // Opens the file and walks the tree; the number of objects, or -1 if libhdf5 fails
    long long tree_walk(const std::string &fn)
    {
        hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
            return -1;
        long long n = 0;
        auto count = [](hid_t, const char *, const H5O_info2_t *, void *n) -> herr_t
        {
            ++*(long long *)n;
            return 0;
        };
        const herr_t status = H5Ovisit3(file, H5_INDEX_NAME, H5_ITER_INC, count, &n, H5O_INFO_BASIC);
        H5Fclose(file);
        return status < 0 ? -1 : n;
    }

    bool link_exists(const std::string &fn, const char *name)
    {
        id_guard file{H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
        return file >= 0 && H5Lexists(file, name, H5P_DEFAULT) > 0;
    }

    // Link names of a group in creation order
    std::vector<std::string> links_by_creation_order(const std::string &fn, const char *group)
    {
        id_guard file{H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
        id_guard g{H5Gopen2(file, group, H5P_DEFAULT), H5Gclose};
        std::vector<std::string> names;
        hsize_t position = 0;
        auto collect = [](hid_t, const char *name, const H5L_info2_t *, void *names) -> herr_t
        {
            ((std::vector<std::string> *)names)->push_back(name);
            return 0;
        };
        CHECK(H5Literate2(g, H5_INDEX_CRT_ORDER, H5_ITER_INC, &position, collect, &names) >= 0);
        return names;
    }

    // Changes the file with the HDF5 C library
    void modify(const std::string &fn, const std::function<void(hid_t)> &edit)
    {
        id_guard file{H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose};
        REQUIRE(file >= 0);
        edit(file);
    }

    // Adds a scalar variable-length string dataset
    void add_vlen_string(hid_t file, const char *path)
    {
        id_guard type{H5Tcopy(H5T_C_S1), H5Tclose}, space{H5Screate(H5S_SCALAR), H5Sclose};
        H5Tset_size(type, H5T_VARIABLE);
        id_guard dset{H5Dcreate2(file, path, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose};
        REQUIRE(dset >= 0);
        const char *text = "variable";
        REQUIRE(H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &text) >= 0);
    }
}

TEST_CASE("hdf5_write - the HDF5 C library counts the attributes of every object")
{
    // Without an Attribute Info message, libhdf5 lists the attributes of a version 2 object header but counts 0
    const std::string fn = "test_hdf5_interop_num_attrs.hdf5";
    size_t n_files = 0, n_objects = 0;
    std::vector<std::string> mismatches;
    reference_file(fn, [&]()
                   { n_objects += check_attribute_counts(fn, mismatches), ++n_files; });
    CHECK(n_files == 10);
    INFO("objects checked " << n_objects);
    CHECK(n_objects > 10);
    CHECK(mismatches == std::vector<std::string>());
    std::remove(fn.c_str());
}

TEST_CASE("hdf5_write - the HDF5 C library reads every intermediate state of an interrupted overwrite")
{
    // File A: 51 slots, slots 0-49 with minimal channels (root link chunk 0 holds IDs 1-41, link chunk 1 IDs 42-50);
    // file B: A after an overwrite of slot 0 (new ID 51)
    const std::string fn_a = "test_hdf5_interop_int_a.hdf5", fn_b = "test_hdf5_interop_int_b.hdf5",
                      fn = "test_hdf5_interop_int.hdf5";
    quadriga_lib::hdf5_create(fn_a, 51);
    for (unsigned s = 0; s < 50; ++s)
    {
        auto c = minimal_channel(s + 1.0);
        quadriga_lib::hdf5_write(&c, fn_a, s);
    }
    const std::vector<unsigned char> A = load(fn_a);
    save(fn_b, A);
    auto c = minimal_channel(100.0);
    quadriga_lib::hdf5_write(&c, fn_b, 0);
    const std::vector<unsigned char> B = load(fn_b);
    const size_t eof_a = eof_address(A), eof_b = eof_address(B);
    REQUIRE(eof_a == A.size());
    REQUIRE(eof_b == B.size());

    // Data ranges of Order and ChannelRef
    std::vector<std::pair<size_t, size_t>> data; // [begin, end)
    {
        id_guard file{H5Fopen(fn_b.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
        for (const char *name : {"ChannelRef", "Order"})
        {
            id_guard dset{H5Dopen2(file, name, H5P_DEFAULT), H5Dclose};
            const size_t offset = (size_t)H5Dget_offset(dset);
            data.push_back({offset, offset + (size_t)H5Dget_storage_size(dset)});
        }
    }

    // The regions in which A and B differ: the superblock, one entry of ChannelRef and Order each, and the two root
    // link chunks (split at the largest gap): the higher one received the new link (step 3), the lower one lost the
    // old link (step 5)
    struct region
    {
        size_t begin = ~(size_t)0, end = 0;
        void add(size_t i) { begin = std::min(begin, i), end = std::max(end, i + 1); }
    };
    region superblock, ref, order, link_low, link_high;
    std::vector<size_t> link_bytes;
    for (size_t i = 0; i < eof_a; ++i)
        if (A[i] != B[i])
        {
            if (i < 48)
                superblock.add(i);
            else if (i >= data[0].first && i < data[0].second)
                ref.add(i);
            else if (i >= data[1].first && i < data[1].second)
                order.add(i);
            else
                link_bytes.push_back(i);
        }
    REQUIRE(link_bytes.size() >= 2);
    size_t split = 1;
    for (size_t k = 1; k < link_bytes.size(); ++k)
        if (link_bytes[k] - link_bytes[k - 1] > link_bytes[split] - link_bytes[split - 1])
            split = k;
    for (size_t k = 0; k < link_bytes.size(); ++k)
        (k < split ? link_low : link_high).add(link_bytes[k]);
    CHECK(ref.end - ref.begin <= 8);
    CHECK(order.end - order.begin <= 4);
    CHECK(link_low.end <= link_high.begin);

    // The states after steps 1, 2, 3, 4a, 4b, 5 of the update order
    region block;
    block.begin = eof_a, block.end = eof_b;
    std::vector<unsigned char> state = A;
    state.resize(B.size());
    const region steps[6] = {block, superblock, link_high, ref, order, link_low};
    for (int i = 0; i < 6; ++i)
    {
        INFO("state after step " << i + 1);
        std::copy(B.begin() + steps[i].begin, B.begin() + steps[i].end, state.begin() + steps[i].begin);
        save(fn, state);
        CHECK(tree_walk(fn) > 0);
        CHECK(link_exists(fn, "channel_1") == (i < 5)); // The old group until step 5
        CHECK(link_exists(fn, "channel_51") == (i >= 2)); // The new group from step 3

        // A read-write session leaves the file unchanged, or truncates it to the end-of-file address
        const size_t eof = eof_address(state);
        {
            id_guard file{H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose};
            CHECK(file >= 0);
        }
        CHECK(load(fn) == std::vector<unsigned char>(state.begin(), state.begin() + eof));
    }
    CHECK(state == B);

    std::remove(fn_a.c_str()), std::remove(fn_b.c_str()), std::remove(fn.c_str());
}

TEST_CASE("hdf5_write - the HDF5 C library finds 8000 root links through the root link directory")
{
    const std::string fn = "test_hdf5_interop_8000.hdf5";
    const unsigned n = 8000;
    quadriga_lib::hdf5_create(fn, n);
    std::vector<unsigned> order(n);
    for (unsigned s = 0; s < n; ++s)
    {
        auto c = minimal_channel(s + 1.0);
        quadriga_lib::hdf5_write(&c, fn, s);
        order[s] = s + 1;
    }
    unsigned next_id = n + 1;
    for (unsigned s : {0u, 3999u, 7999u, 7999u, 7999u})
    {
        auto c = minimal_channel(100000.0 + next_id);
        quadriga_lib::hdf5_write(&c, fn, s);
        order[s] = next_id++;
    }

    std::set<std::string> expected = {"ChannelDims", "ChannelRef", "Order", "Version"};
    for (unsigned id : order)
        expected.insert("channel_" + std::to_string(id));
    {
        id_guard file{H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
        REQUIRE(file >= 0);
        std::set<std::string> names;
        hsize_t position = 0;
        auto collect = [](hid_t, const char *name, const H5L_info2_t *, void *names) -> herr_t
        {
            ((std::set<std::string> *)names)->insert(name);
            return 0;
        };
        CHECK(H5Literate2(file, H5_INDEX_NAME, H5_ITER_NATIVE, &position, collect, &names) >= 0);
        CHECK(names == expected);
        for (const char *name : {"channel_2", "channel_3999", "channel_8005"})
        {
            INFO("group " << name);
            id_guard group{H5Oopen(file, name, H5P_DEFAULT), H5Oclose};
            CHECK(group >= 0);
        }
    }
    check_channel_ref(fn, order);
    std::remove(fn.c_str());
}

TEST_CASE("hdf5 - files modified with the HDF5 C library are rejected or read consistently")
{
    const std::string fn = "test_hdf5_interop_modified.hdf5";
    std::any x = 1.0;
    std::vector<std::string> names;
    auto names_of = [&](unsigned ix, unsigned iy)
    {
        names.clear();
        quadriga_lib::hdf5_read_dset_names(fn, &names, ix, iy);
        return names;
    };
    // Calls that throw the "Unsupported" message and leave the file unchanged
    auto check_rejected = [&](std::vector<std::pair<const char *, std::function<void()>>> calls)
    {
        const std::vector<unsigned char> before = load(fn);
        for (auto &w : calls)
        {
            INFO("function " << w.first);
            CHECK_THROWS_MATCHES(w.second(), std::invalid_argument, Catch::Matchers::Message(unsupported_message));
            CHECK(load(fn) == before);
        }
    };

    SECTION("a group added to a compact root: read functions work, writers are rejected")
    {
        std::remove(fn.c_str());
        quadriga_lib::hdf5_create(fn, 4);
        for (unsigned s = 0; s < 3; ++s)
        {
            auto c = minimal_channel(s + 1.0);
            quadriga_lib::hdf5_write(&c, fn, s);
        }
        modify(fn, [](hid_t file)
               { id_guard g{H5Gcreate2(file, "/extra", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose}; REQUIRE(g >= 0); });
        arma::u32_vec id;
        CHECK(arma::all(quadriga_lib::hdf5_read_layout(fn, &id) == arma::u32_vec({4, 1, 1, 1})));
        CHECK(arma::all(id == arma::u32_vec({1, 2, 3, 0})));
        for (unsigned s = 0; s < 3; ++s)
            CHECK(quadriga_lib::hdf5_read_channel<double>(fn, s).coeff_re[0](0) == s + 1.0);
        auto c = minimal_channel(9.0);
        check_rejected({{"hdf5_write", [&]()
                                 { quadriga_lib::hdf5_write(&c, fn, 3); }},
                                {"hdf5_write overwrite", [&]()
                                 { quadriga_lib::hdf5_write(&c, fn, 0); }},
                                {"hdf5_write_dset, empty slot", [&]()
                                 { quadriga_lib::hdf5_write_dset(fn, "x", &x, 3); }},
                                {"hdf5_write_dset, filled slot", [&]()
                                 { quadriga_lib::hdf5_write_dset(fn, "x", &x, 0); }},
                                {"hdf5_reshape_layout", [&]()
                                 { quadriga_lib::hdf5_reshape_layout(fn, 2, 2); }}});
        quadriga_lib::hdf5_create(fn, 2); // Replaces the file
        CHECK(quadriga_lib::hdf5_write(&c, fn, 1) == 0);
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 1).coeff_re[0](0) == 9.0);
    }
    SECTION("a group added to the root of the reference file (dense root): every function is rejected")
    {
        reference_file(fn);
        modify(fn, [](hid_t file)
               { id_guard g{H5Gcreate2(file, "/extra", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose}; REQUIRE(g >= 0); });
        auto c = minimal_channel(9.0);
        check_rejected({{"hdf5_read_layout", [&]()
                                 { quadriga_lib::hdf5_read_layout(fn); }},
                                {"hdf5_read_channel", [&]()
                                 { quadriga_lib::hdf5_read_channel<double>(fn, 0); }},
                                {"hdf5_read_dset", [&]()
                                 { quadriga_lib::hdf5_read_dset(fn, "note", 0, 1); }},
                                {"hdf5_read_dset_names", [&]()
                                 { names_of(0, 1); }},
                                {"hdf5_write", [&]()
                                 { quadriga_lib::hdf5_write(&c, fn, 1, 2); }},
                                {"hdf5_write_dset", [&]()
                                 { quadriga_lib::hdf5_write_dset(fn, "x", &x, 0); }},
                                {"hdf5_reshape_layout", [&]()
                                 { quadriga_lib::hdf5_reshape_layout(fn, 6); }}});
    }
    SECTION("a variable-length string in a compact 'Data' group: listed, reading it is rejected")
    {
        reference_file(fn);
        modify(fn, [](hid_t file)
               { add_vlen_string(file, "/channel_3/Data/par_vlen"); });
        CHECK(names_of(0, 1) == std::vector<std::string>({"note", "values", "vlen"}));
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset(fn, "vlen", 0, 1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn, 0, 1), std::invalid_argument);
        CHECK(std::any_cast<std::string>(quadriga_lib::hdf5_read_dset(fn, "note", 0, 1)) == "note");
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 0, 0).par_names.size() == 10);
    }
    SECTION("a variable-length string in a dense 'Data' group: that slot is rejected, the others work")
    {
        reference_file(fn);
        modify(fn, [](hid_t file)
               { add_vlen_string(file, "/channel_1/Data/par_vlen"); });
        check_rejected({{"hdf5_read_channel", [&]()
                                 { quadriga_lib::hdf5_read_channel<double>(fn, 0, 0); }},
                                {"hdf5_read_dset", [&]()
                                 { quadriga_lib::hdf5_read_dset(fn, "string", 0, 0); }},
                                {"hdf5_read_dset_names", [&]()
                                 { names_of(0, 0); }},
                                {"hdf5_write_dset", [&]()
                                 { quadriga_lib::hdf5_write_dset(fn, "x", &x, 0, 0); }}});
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 1, 0).n_snap() == 1);
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 1, 1).name == "new");
        CHECK(names_of(0, 1) == std::vector<std::string>({"note", "values"}));
        CHECK(names_of(0, 2) == std::vector<std::string>({"added"}));

        // An overwrite does not read the old group: it replaces the slot, and both libraries read the result
        auto c = minimal_channel(9.0);
        CHECK(quadriga_lib::hdf5_write(&c, fn, 0, 0) == 1);
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 0, 0).coeff_re[0](0) == 9.0);
        CHECK(tree_walk(fn) > 0);
        CHECK(!link_exists(fn, "channel_1"));
        CHECK(link_exists(fn, "channel_7"));
    }
    SECTION("a dataset deleted from 'Data': the others are read and new ones appended")
    {
        reference_file(fn);
        modify(fn, [](hid_t file)
               { REQUIRE(H5Ldelete(file, "/channel_3/Data/par_note", H5P_DEFAULT) >= 0); });
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 0, 1).par_names == std::vector<std::string>({"values"}));
        CHECK(!quadriga_lib::hdf5_read_dset(fn, "note", 0, 1).has_value());
        CHECK(names_of(0, 1) == std::vector<std::string>({"values"}));
        quadriga_lib::hdf5_write_dset(fn, "x", &x, 0, 1);
        CHECK(names_of(0, 1) == std::vector<std::string>({"values", "x"}));
        CHECK(std::any_cast<double>(quadriga_lib::hdf5_read_dset(fn, "x", 0, 1)) == 1.0);
        CHECK(tree_walk(fn) > 0);
        CHECK(links_by_creation_order(fn, "/channel_3/Data") == std::vector<std::string>({"par_values", "par_x"}));
    }
    SECTION("a file written by the HDF5 C library with default settings (superblock version 0)")
    {
        {
            id_guard file{H5Fcreate(fn.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose};
            REQUIRE(file >= 0);
        }
        auto c = minimal_channel(9.0);
        check_rejected({{"hdf5_read_layout", [&]()
                                 { quadriga_lib::hdf5_read_layout(fn); }},
                                {"hdf5_read_channel", [&]()
                                 { quadriga_lib::hdf5_read_channel<double>(fn); }},
                                {"hdf5_read_dset", [&]()
                                 { quadriga_lib::hdf5_read_dset(fn, "x"); }},
                                {"hdf5_read_dset_names", [&]()
                                 { names_of(0, 0); }},
                                {"hdf5_write", [&]()
                                 { quadriga_lib::hdf5_write(&c, fn); }},
                                {"hdf5_write_dset", [&]()
                                 { quadriga_lib::hdf5_write_dset(fn, "x", &x); }},
                                {"hdf5_reshape_layout", [&]()
                                 { quadriga_lib::hdf5_reshape_layout(fn, 1); }}});
    }
    std::remove(fn.c_str());
}
