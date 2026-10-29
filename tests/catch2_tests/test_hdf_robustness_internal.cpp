// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// The public HDF5 channel-file functions on damaged files and on files that are valid HDF5 but outside the subset
// written by the library: corrupted superblock, root, index and object headers, inconsistent channel data, root link
// structures and growable groups changed by other tools, a sweep of single-byte flips over all headers, the
// intermediate states of interrupted writes, and a file whose root links fill the directory. Round trips of valid
// files are tested in test_hdf_functions.cpp, the view of the HDF5 C library in test_hdf5_interop.cpp, the format
// layer itself in test_qd_hdf5_internal.cpp.
//
// Oracle: the documented behavior. A malformed file gives std::invalid_argument (with the documented message where
// one is documented), never another exception, a crash, or an allocation larger than the file; a writer that throws
// leaves the file byte-identical; accepted variants and interrupted writes read back the old or the new content.
// The private header qd_hdf5.hpp is used only as a patch tool: to locate objects through their links, and to rewrite
// a changed object-header chunk with a valid checksum, so that the checks behind the checksums are reached.
//
// Fixtures: files are written with the public API (channels with distinct values per slot), then patched. Each test
// case uses its own file names and removes its files at the end. For the sanitizer build, run with
// ASAN_OPTIONS=max_allocation_size_mb=1024:allocator_may_return_null=0 so that an allocation beyond 1 GB fails.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_lib.hpp"
#include "qd_hdf5.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Diagnostic output: off by default. QD_DIAG_HDF=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way.
#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_HDF")

namespace h5 = qd::hdf5;
using Catch::Matchers::Message;

namespace
{
    using bytes = std::vector<uint8_t>;

    const std::string unsupported_message =
        "Unsupported HDF5 file: quadriga-lib reads only channel files of version 3 that it wrote itself; files written "
        "by quadriga-lib 0.12.4 and earlier (version 2) and files modified with other HDF5 tools are not supported. "
        "Version-2 files can be converted with tools/hdf5_convert_v2.py.";
    const std::string corrupted_message = "Storage index in HDF file is corrupted.";

    // Removes the files when the test case starts and when it ends
    struct test_files
    {
        std::vector<std::string> fn;
        explicit test_files(std::vector<std::string> names) : fn(std::move(names)) { remove_all(); }
        ~test_files() { remove_all(); }
        void remove_all()
        {
            for (const auto &f : fn)
                std::remove(f.c_str());
        }
    };

    // ---- Channels ----

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

    // Coefficient of a minimal channel at a slot; -1 for an empty slot
    double value_of(const std::string &fn, unsigned slot)
    {
        const auto c = quadriga_lib::hdf5_read_channel<double>(fn, slot);
        return c.coeff_re.empty() ? -1.0 : c.coeff_re[0](0);
    }

    template <typename A>
    A filled(A x, double start)
    {
        x.imbue([&]()
                { return (typename A::elem_type)(start += 0.25); });
        return x;
    }

    // Every structured field, n_rx = 2, n_tx = 3, one snapshot per entry of n_path
    quadriga_lib::channel<double> full_channel(const std::vector<arma::uword> &n_path)
    {
        const arma::uword n_snap = n_path.size();
        quadriga_lib::channel<double> c;
        c.name = "full";
        c.initial_position = 1;
        c.center_frequency = filled(arma::vec(n_snap), 1.0);
        c.tx_pos = filled(arma::mat(3, n_snap), 1.0);
        c.rx_pos = filled(arma::mat(3, n_snap), 10.0);
        c.tx_orientation = filled(arma::mat(3, 1), -1.0);
        for (arma::uword s = 0; s < n_snap; ++s)
        {
            const arma::uword np = n_path[s];
            const double o = 100.0 * (double)s;
            c.coeff_re.push_back(filled(arma::cube(2, 3, np), o));
            c.coeff_im.push_back(filled(arma::cube(2, 3, np), o + 10.0));
            c.delay.push_back(filled(arma::cube(2, 3, np), o + 20.0));
            c.path_gain.push_back(filled(arma::vec(np), o + 30.0));
            c.path_length.push_back(filled(arma::vec(np), o + 40.0));
            c.path_polarization.push_back(filled(arma::mat(8, np), o + 50.0));
            c.path_angles.push_back(filled(arma::mat(np, 4), o + 60.0));
            c.path_fbs_pos.push_back(filled(arma::mat(3, np), o + 70.0));
            c.path_lbs_pos.push_back(filled(arma::mat(3, np), o + 80.0));
            arma::u32_vec no_interact(np);
            for (arma::uword p = 0; p < np; ++p)
                no_interact(p) = (unsigned)(p % 3);
            c.no_interact.push_back(no_interact);
            c.interact_coord.push_back(filled(arma::mat(3, arma::accu(no_interact)), o + 90.0));
        }
        return c;
    }

    // ---- Raw file access ----

    bytes load(const std::string &fn)
    {
        std::ifstream f(fn, std::ios::binary);
        return bytes(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    void save(const std::string &fn, const bytes &b)
    {
        std::ofstream f(fn, std::ios::binary | std::ios::trunc);
        f.write((const char *)b.data(), (std::streamsize)b.size());
    }

    void put(const std::string &fn, uint64_t addr, const bytes &b)
    {
        h5::file f;
        REQUIRE(f.open(fn, "r+b"));
        f.write(addr, b.data(), b.size());
        f.close();
    }

    uint64_t eof_of(const std::string &fn)
    {
        h5::file f;
        REQUIRE(f.open(fn, "rb"));
        return h5::read_superblock(f).eof;
    }

    // Checksum of a chunk image or the superblock: lookup3 of all bytes before the last four
    void resign(bytes &image)
    {
        h5::put_le(&image[image.size() - 4], h5::lookup3(image.data(), image.size() - 4), 4);
    }

    // Changes the superblock (48 bytes) and writes it with a new checksum
    void patch_superblock(const std::string &fn, const std::function<void(bytes &)> &edit)
    {
        bytes sb = load(fn);
        sb.resize(48);
        edit(sb);
        resign(sb);
        put(fn, 0, sb);
    }

    // Appends bytes at the end-of-file address (moved behind them); returns their address
    uint64_t append(const std::string &fn, const bytes &b)
    {
        h5::file f;
        REQUIRE(f.open(fn, "r+b"));
        const h5::superblock sb = h5::read_superblock(f);
        f.write(sb.eof, b.data(), b.size());
        const bytes head = h5::superblock_bytes(sb.eof + b.size(), sb.root);
        f.write(0, head.data(), head.size());
        f.close();
        return sb.eof;
    }

    // ---- Objects ----

    h5::object object_at(const std::string &fn, uint64_t addr)
    {
        h5::file f;
        REQUIRE(f.open(fn, "rb"));
        const h5::superblock sb = h5::read_superblock(f);
        return h5::read_object(f, addr, sb.eof);
    }

    // Address of the object at 'path' ("channel_1/Snap_0/delay"; "" = root group), found through the links
    uint64_t locate(const std::string &fn, const std::string &path)
    {
        h5::file f;
        REQUIRE(f.open(fn, "rb"));
        const h5::superblock sb = h5::read_superblock(f);
        uint64_t addr = sb.root;
        for (size_t pos = 0; pos < path.size();)
        {
            const size_t end = std::min(path.find('/', pos), path.size());
            const std::string name = path.substr(pos, end - pos);
            const h5::object o = h5::read_object(f, addr, sb.eof);
            const h5::group g = h5::decode_group(o);
            const h5::link *l = g.find(name);
            INFO("path " << path);
            REQUIRE(l != nullptr);
            addr = l->target;
            pos = end + 1;
        }
        return addr;
    }

    // Data address and element count of a contiguous dataset
    std::pair<uint64_t, uint64_t> contiguous_data(const std::string &fn, const std::string &path)
    {
        const h5::object o = object_at(fn, locate(fn, path));
        const h5::dataset d = h5::decode_dataset(o, eof_of(fn));
        REQUIRE(!d.compact);
        return {d.address, d.n_elem};
    }

    // One message of an object header: type, flags and data
    struct message
    {
        uint8_t type, flags;
        bytes data;
    };

    std::vector<message> messages_of(const h5::chunk &c)
    {
        std::vector<message> m;
        for (const h5::message_ref &r : c.messages)
            if (r.type != h5::msg_nil)
                m.push_back({r.type, r.flags, bytes(c.data(r), c.data(r) + r.size)});
        return m;
    }

    bytes encode(const std::vector<message> &msgs)
    {
        bytes out;
        for (const message &m : msgs)
        {
            out.push_back(m.type);
            h5::put_le(out, m.data.size(), 2);
            out.push_back(m.flags);
            out.insert(out.end(), m.data.begin(), m.data.end());
        }
        return out;
    }

    // Name of a Link or Attribute message
    std::string name_of(const message &m)
    {
        if (m.type == h5::msg_link)
        {
            h5::link l;
            if (h5::decode_link(m.data.data(), m.data.size(), l))
                return std::string(l.name);
        }
        if (m.type == h5::msg_attribute)
        {
            h5::attribute a;
            if (h5::decode_attribute(m.data.data(), m.data.size(), a))
                return std::string(a.name);
        }
        return "";
    }

    // Index of the first message of a type (and, for links and attributes, a name)
    size_t find(const std::vector<message> &msgs, uint8_t type, const std::string &name = "")
    {
        for (size_t i = 0; i < msgs.size(); ++i)
            if (msgs[i].type == type && (name.empty() || name_of(msgs[i]) == name))
                return i;
        INFO("message type " << (int)type << " '" << name << "'");
        FAIL("message not found");
        return 0;
    }

    message link_message(const std::string &name, uint64_t target)
    {
        bytes b;
        h5::link_message(b, name, target);
        return {h5::msg_link, 0, bytes(b.begin() + 4, b.end())};
    }

    message continuation_message(uint64_t addr, uint64_t length)
    {
        message m{h5::msg_continuation, 0, {}};
        h5::put_le(m.data, addr, 8);
        h5::put_le(m.data, length, 8);
        return m;
    }

    message dataspace_message(std::vector<uint64_t> dims, uint8_t type = 1)
    {
        message m{h5::msg_dataspace, 0, {2, (uint8_t)dims.size(), 0, type}};
        for (uint64_t d : dims)
            h5::put_le(m.data, d, 8);
        return m;
    }

    message attribute_info_message()
    {
        message m{h5::msg_attribute_info, 0x04, {0, 0}};
        h5::put_le(m.data, h5::undef, 8);
        h5::put_le(m.data, h5::undef, 8);
        return m;
    }

    // Redirects the link at 'path' (and the ChannelRef entries that point to the same object) to 'target'
    void redirect(const std::string &fn, const std::string &path, uint64_t target)
    {
        const size_t slash = path.rfind('/');
        const std::string parent = slash == std::string::npos ? "" : path.substr(0, slash);
        const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
        const uint64_t old = locate(fn, path);
        h5::object o = object_at(fn, locate(fn, parent));
        for (h5::chunk &c : o.chunks)
            for (const h5::message_ref &r : c.messages)
            {
                h5::link l;
                if (r.type == h5::msg_link && h5::decode_link(c.data(r), r.size, l) && l.name == name)
                {
                    bytes image = c.image;
                    h5::put_le(&image[r.offset + r.size - 8], target, 8);
                    resign(image);
                    put(fn, c.addr, image);
                }
            }
        if (!parent.empty())
            return;
        const auto ref = contiguous_data(fn, "ChannelRef");
        bytes entries = load(fn);
        for (uint64_t i = 0; i < ref.second; ++i)
            if (h5::get_le(&entries[ref.first + 8 * i], 8) == old)
            {
                bytes entry(8);
                h5::put_le(entry.data(), target, 8);
                put(fn, ref.first + 8 * i, entry);
            }
    }

    // Replaces the messages of chunk k of the object at 'path'. If they fit, the chunk is rewritten in place (tail
    // rule, valid checksum); otherwise chunk 0 of a single-chunk object is written as a new header at the end of the
    // file and the links to it are redirected.
    void edit_messages(const std::string &fn, const std::string &path, size_t k,
                       const std::function<void(std::vector<message> &)> &edit)
    {
        h5::object o = object_at(fn, locate(fn, path));
        REQUIRE(k < o.chunks.size());
        h5::chunk &c = o.chunks[k];
        std::vector<message> msgs = messages_of(c);
        edit(msgs);
        const bytes encoded = encode(msgs);
        if (encoded.size() <= c.body())
        {
            h5::rewrite_chunk(c, encoded);
            put(fn, c.addr, c.image);
            return;
        }
        REQUIRE((k == 0 && o.chunks.size() == 1 && !path.empty()));
        bytes image;
        h5::encode_chunk(image, true, encoded.data(), encoded.size(), encoded.size());
        redirect(fn, path, append(fn, image));
    }

    // Rewrites chunk 0 of the object at 'path' with its messages and a body 'shrink' bytes smaller (the free space
    // shrinks; the bytes after the shorter chunk are no longer part of it)
    void shrink_chunk0(const std::string &fn, const std::string &path, size_t shrink)
    {
        const h5::object o = object_at(fn, locate(fn, path));
        const h5::chunk &c = o.chunks[0];
        REQUIRE(c.free_space >= shrink);
        const bytes msgs = encode(messages_of(c));
        bytes image;
        h5::encode_chunk(image, true, msgs.data(), msgs.size(), c.body() - shrink, c.image[5] & 0x03);
        put(fn, c.addr, image);
    }

    // Changes the raw image of chunk k of the object at 'addr' (signature to checksum), optionally re-signed
    void edit_chunk(const std::string &fn, uint64_t addr, size_t k,
                    const std::function<void(bytes &, const h5::chunk &)> &edit, bool re_sign = true)
    {
        const h5::object o = object_at(fn, addr);
        REQUIRE(k < o.chunks.size());
        bytes image = o.chunks[k].image;
        edit(image, o.chunks[k]);
        if (re_sign)
            resign(image);
        put(fn, o.chunks[k].addr, image);
    }

    // Position of a message (by type and name) in a chunk image; header at offset - 4
    const h5::message_ref &message_in(const h5::chunk &c, uint8_t type, const std::string &name = "")
    {
        for (const h5::message_ref &r : c.messages)
            if (r.type == type && (name.empty() || name_of({r.type, r.flags, bytes(c.data(r), c.data(r) + r.size)}) == name))
                return r;
        FAIL("message not found");
        return c.messages[0];
    }

    // ---- Calls ----

    // A named call of a public function
    struct call
    {
        const char *name;
        std::function<void()> run;
    };

    // Every public function on the standard file (below)
    std::vector<call> all_functions(const std::string &fn)
    {
        return {{"hdf5_read_layout", [=]()
                 { arma::u32_vec id; quadriga_lib::hdf5_read_layout(fn, &id); }},
                {"hdf5_read_channel", [=]()
                 { quadriga_lib::hdf5_read_channel<double>(fn, 0); }},
                {"hdf5_read_dset", [=]()
                 { quadriga_lib::hdf5_read_dset(fn, "x", 0); }},
                {"hdf5_read_dset_names", [=]()
                 { std::vector<std::string> names; quadriga_lib::hdf5_read_dset_names(fn, &names, 0); }},
                {"hdf5_write", [=]()
                 { auto c = minimal_channel(5.0); quadriga_lib::hdf5_write(&c, fn, 1); }},
                {"hdf5_write_dset", [=]()
                 { std::any x = 5.0; quadriga_lib::hdf5_write_dset(fn, "new", &x, 0); }},
                {"hdf5_reshape_layout", [=]()
                 { quadriga_lib::hdf5_reshape_layout(fn, 4); }}};
    }

    // Every call throws std::invalid_argument (with 'msg' if not empty) and leaves the file unchanged
    void check_all_throw(const std::string &fn, const std::vector<call> &calls, const std::string &msg = "")
    {
        const bytes before = load(fn);
        for (const call &c : calls)
        {
            INFO("function " << c.name);
            if (msg.empty())
                CHECK_THROWS_AS(c.run(), std::invalid_argument);
            else
                CHECK_THROWS_MATCHES(c.run(), std::invalid_argument, Message(msg));
            CHECK(load(fn) == before);
        }
    }

    // ---- Standard file ----
    // 4 slots: 0 = 1 snapshot, 4 RX, 2 TX, 50 paths, with unstructured data x (double), s (string), u (uint32 [3]),
    // big (double [1000], contiguous); 1 = 3 snapshots with every field (np 3, 0, 2); 2 = created by hdf5_write_dset
    // (dataset a); 3 = empty. Channel IDs 1, 2, 3.
    void standard_file(const std::string &fn)
    {
        std::remove(fn.c_str());
        quadriga_lib::hdf5_create(fn, 4);
        quadriga_lib::channel<double> c0;
        c0.name = "test";
        c0.center_frequency = {3e9};
        c0.tx_pos = arma::mat(3, 1, arma::fill::ones);
        c0.rx_pos = arma::mat(3, 1, arma::fill::zeros);
        c0.coeff_re.push_back(arma::cube(4, 2, 50, arma::fill::value(0.5)));
        c0.coeff_im.push_back(arma::cube(4, 2, 50, arma::fill::value(-0.5)));
        c0.delay.push_back(arma::cube(4, 2, 50, arma::fill::value(1e-7)));
        c0.par_names = {"x", "s", "u", "big"};
        c0.par_data = {1.5, std::string("abc"), arma::Col<unsigned>{1, 2, 3}, arma::regspace<arma::vec>(1.0, 1000.0)};
        quadriga_lib::hdf5_write(&c0, fn, 0);
        auto c1 = full_channel({3, 0, 2});
        quadriga_lib::hdf5_write(&c1, fn, 1);
        std::any a = 2.5;
        quadriga_lib::hdf5_write_dset(fn, "a", &a, 2);
    }

    // The interrupted-write states: A plus the regions of B, applied in order; the last state must equal B
    struct region
    {
        uint64_t addr, size;
    };

    std::vector<bytes> interrupted_states(const bytes &A, const bytes &B, const std::vector<region> &regions)
    {
        std::vector<bytes> states;
        bytes s = A;
        for (const region &r : regions)
        {
            REQUIRE(r.addr + r.size <= B.size());
            if (s.size() < r.addr + r.size)
                s.resize(r.addr + r.size);
            std::copy(B.begin() + r.addr, B.begin() + r.addr + r.size, s.begin() + r.addr);
            states.push_back(s);
        }
        CHECK(states.back() == B);
        return states;
    }

    // Root link chunk k (counted from 0) of a file without directory chunks: address and length
    region link_chunk(const std::string &fn, size_t k)
    {
        const h5::object o = object_at(fn, locate(fn, ""));
        REQUIRE(k + 1 < o.chunks.size());
        return {o.chunks[k + 1].addr, o.chunks[k + 1].length()};
    }

    // Channel IDs linked in root link chunk k
    std::vector<long long> link_chunk_ids(const std::string &fn, size_t k)
    {
        const h5::object o = object_at(fn, locate(fn, ""));
        REQUIRE(k + 1 < o.chunks.size());
        std::vector<long long> ids;
        for (const message &m : messages_of(o.chunks[k + 1]))
            ids.push_back(h5::parse_decimal(name_of(m).substr(8)));
        return ids;
    }

    std::vector<std::string> dset_names(const std::string &fn, unsigned slot)
    {
        std::vector<std::string> names;
        quadriga_lib::hdf5_read_dset_names(fn, &names, slot);
        return names;
    }
}

TEST_CASE("hdf5 - damaged superblocks are rejected")
{
    const std::string fn = "test_hdf_rb_superblock.hdf5";
    test_files files({fn});
    standard_file(fn);
    const bytes good = load(fn);
    const auto calls = all_functions(fn);

    SECTION("empty file and 7-byte file")
    {
        save(fn, {});
        check_all_throw(fn, calls, "Not an HDF5 file.");
        save(fn, bytes(good.begin(), good.begin() + 7));
        check_all_throw(fn, calls, "Not an HDF5 file.");
    }
    SECTION("wrong signature byte")
    {
        bytes b = good;
        b[1] = 'h';
        save(fn, b);
        check_all_throw(fn, calls, "Not an HDF5 file.");
    }
    SECTION("superblock checksum mismatch")
    {
        bytes b = good;
        b[30] ^= 0x01; // End-of-file address
        save(fn, b);
        check_all_throw(fn, calls);
    }
    SECTION("superblock version 0")
    {
        patch_superblock(fn, [](bytes &sb)
                         { sb[8] = 0; });
        check_all_throw(fn, calls, unsupported_message);
    }
    SECTION("end-of-file address beyond the file size")
    {
        save(fn, bytes(good.begin(), good.end() - 100));
        check_all_throw(fn, calls);
    }
    SECTION("root address beyond the end of the file")
    {
        patch_superblock(fn, [&](bytes &sb)
                         { h5::put_le(&sb[36], good.size() + 8, 8); });
        check_all_throw(fn, calls);
    }
    SECTION("superblock extension address 0")
    {
        patch_superblock(fn, [](bytes &sb)
                         { h5::put_le(&sb[20], 0, 8); });
        check_all_throw(fn, calls, unsupported_message);
    }
}

TEST_CASE("hdf5 - files with the consistency flag or bytes after the end of file are read and extended")
{
    const std::string fn = "test_hdf_rb_accepted.hdf5";
    test_files files({fn});
    standard_file(fn);

    auto check_contents = [&]()
    {
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 0).coeff_re[0](0, 0, 49) == 0.5);
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 1).n_snap() == 3);
        CHECK(std::any_cast<double>(quadriga_lib::hdf5_read_dset(fn, "a", 2)) == 2.5);
        CHECK(std::any_cast<std::string>(quadriga_lib::hdf5_read_dset(fn, "s", 0)) == "abc");
    };

    SECTION("consistency flag bit 0 (left by a crashed read-write session of the HDF5 library)")
    {
        patch_superblock(fn, [](bytes &sb)
                         { sb[11] = 1; });
        arma::u32_vec id;
        CHECK(arma::all(quadriga_lib::hdf5_read_layout(fn, &id) == arma::u32_vec({4, 1, 1, 1})));
        CHECK(arma::all(id == arma::u32_vec({1, 2, 3, 0})));
        check_contents();
        auto c = minimal_channel(7.0);
        CHECK(quadriga_lib::hdf5_write(&c, fn, 3) == 0);
        std::any y = 3.5;
        quadriga_lib::hdf5_write_dset(fn, "y", &y, 1);
        CHECK(value_of(fn, 3) == 7.0);
        CHECK(std::any_cast<double>(quadriga_lib::hdf5_read_dset(fn, "y", 1)) == 3.5);
        check_contents();
        quadriga_lib::hdf5_reshape_layout(fn, 2, 2);
        CHECK(arma::all(quadriga_lib::hdf5_read_layout(fn, &id) == arma::u32_vec({2, 2, 1, 1})));
        CHECK(arma::all(id == arma::u32_vec({1, 2, 3, 4})));
    }
    SECTION("4 KB of garbage after the end of file")
    {
        const uint64_t eof = eof_of(fn);
        bytes b = load(fn);
        b.insert(b.end(), 4096, 0xa5);
        save(fn, b);
        check_contents();
        auto c = minimal_channel(7.0);
        CHECK(quadriga_lib::hdf5_write(&c, fn, 3) == 0);
        const uint64_t new_eof = eof_of(fn);
        const uint64_t ref3 = h5::get_le(&load(fn)[contiguous_data(fn, "ChannelRef").first + 24], 8);
        CHECK(ref3 > eof); // The block starts at the old end-of-file address
        CHECK(ref3 < new_eof);
        CHECK(load(fn).size() == eof + 4096); // The block was smaller than the garbage
        check_contents();
        CHECK(value_of(fn, 3) == 7.0);
    }
}

TEST_CASE("hdf5 - a damaged version or storage index and bad channel references are rejected")
{
    const std::string fn = "test_hdf_rb_index.hdf5";
    test_files files({fn});
    standard_file(fn);
    const auto calls = all_functions(fn);
    const uint64_t eof = eof_of(fn);

    auto set_version = [&](unsigned v)
    {
        edit_messages(fn, "Version", 0, [&](std::vector<message> &m)
                      { h5::put_le(&m[find(m, h5::msg_layout)].data[4], v, 4); });
    };
    auto set_dims = [&](std::vector<unsigned> dims)
    {
        bytes b(16);
        for (int i = 0; i < 4; ++i)
            h5::put_le(&b[4 * i], dims[i], 4);
        put(fn, contiguous_data(fn, "ChannelDims").first, b);
    };
    auto set_channel_ref = [&](unsigned slot, uint64_t target)
    {
        bytes b(8);
        h5::put_le(b.data(), target, 8);
        put(fn, contiguous_data(fn, "ChannelRef").first + 8 * slot, b);
    };
    // Functions that read the channel group of slot 2 throw; the others work
    auto check_slot2_rejected = [&]()
    {
        const bytes before = load(fn);
        std::any x = 1.0;
        std::vector<std::string> names;
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn, 2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset(fn, "a", 2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset_names(fn, &names, 2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(fn, "b", &x, 2), std::invalid_argument);
        CHECK(load(fn) == before);
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 0).n_path()(0) == 50);
    };

    SECTION("version 2 and 4")
    {
        set_version(2);
        check_all_throw(fn, calls, "Unsupported channel file version 2; this version of quadriga-lib reads version 3 only.");
        set_version(4);
        check_all_throw(fn, calls, "Unsupported channel file version 4; this version of quadriga-lib reads version 3 only.");
    }
    SECTION("root link 'Version' removed")
    {
        edit_messages(fn, "", 0, [](std::vector<message> &m)
                      { m.erase(m.begin() + find(m, h5::msg_link, "Version")); });
        check_all_throw(fn, calls, unsupported_message);
    }
    SECTION("ChannelDims {3, 1, 1, 1} with 4 slots")
    {
        set_dims({3, 1, 1, 1});
        check_all_throw(fn, calls, corrupted_message);
    }
    SECTION("ChannelDims with more than 2^32 - 1 slots")
    {
        set_dims({65536, 65536, 1, 1});
        check_all_throw(fn, calls, corrupted_message);
    }
    SECTION("ChannelDims entry 0")
    {
        set_dims({4, 1, 0, 1});
        check_all_throw(fn, calls, corrupted_message);
    }
    SECTION("Order data beyond the end of file")
    {
        edit_messages(fn, "Order", 0, [&](std::vector<message> &m)
                      { h5::put_le(&m[find(m, h5::msg_layout)].data[2], eof, 8); });
        check_all_throw(fn, calls);
    }
    SECTION("contiguous data at UNDEF with 16 bytes, and at 2^63 with 2^63 bytes")
    {
        edit_messages(fn, "ChannelDims", 0, [](std::vector<message> &m)
                      { h5::put_le(&m[find(m, h5::msg_layout)].data[2], h5::undef, 8); });
        check_all_throw(fn, calls);
        standard_file(fn);
        edit_messages(fn, "Order", 0, [](std::vector<message> &m)
                      {
                          m[find(m, h5::msg_dataspace)] = dataspace_message({1ULL << 61}); // 2^63 bytes of uint32
                          bytes &layout = m[find(m, h5::msg_layout)].data;
                          h5::put_le(&layout[2], 1ULL << 63, 8);
                          h5::put_le(&layout[10], 1ULL << 63, 8); });
        check_all_throw(fn, calls);
    }
    SECTION("ChannelRef beyond the end of file")
    {
        set_channel_ref(2, eof + 8);
        check_slot2_rejected();
    }
    SECTION("ChannelRef to a dataset header")
    {
        set_channel_ref(2, locate(fn, "channel_1/coeff_re"));
        check_slot2_rejected();
    }
    SECTION("ChannelRef into raw data")
    {
        set_channel_ref(2, contiguous_data(fn, "channel_1/Data/par_big").first + 64);
        check_slot2_rejected();
    }
    SECTION("ChannelRef UNDEF")
    {
        set_channel_ref(2, h5::undef);
        check_slot2_rejected();
    }
    SECTION("ChannelRef 0 is an empty slot")
    {
        set_channel_ref(2, 0);
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 2).empty());
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 2).par_names.empty());
        CHECK(!quadriga_lib::hdf5_read_dset(fn, "a", 2).has_value());
        CHECK(quadriga_lib::hdf5_read_dset_names(fn, nullptr, 2) == 0);
    }
}

TEST_CASE("hdf5 - malformed object headers are rejected")
{
    const std::string fn = "test_hdf_rb_headers.hdf5";
    test_files files({fn});
    standard_file(fn);
    const uint64_t eof = eof_of(fn);
    const uint64_t channel = locate(fn, "channel_1");

    // Readers of slot 0 throw; slot 1 still reads
    auto check_rejected = [&]()
    {
        const bytes before = load(fn);
        std::vector<std::string> names;
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn, 0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<float>(fn, 0), std::invalid_argument);
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 1).n_snap() == 3);
        CHECK(load(fn) == before);
    };
    // Readers of the unstructured dataset 'name' of slot 0 throw; name "" = the group 'Data' itself is damaged, so
    // all its readers throw
    auto check_data_rejected = [&](const std::string &name)
    {
        std::vector<std::string> names;
        check_rejected();
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset(fn, name.empty() ? "x" : name, 0), std::invalid_argument);
        if (name.empty())
            CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset_names(fn, &names, 0), std::invalid_argument);
        else
            CHECK(quadriga_lib::hdf5_read_dset_names(fn, &names, 0) == 4); // The link targets are not opened
    };

    SECTION("chunk 0 size beyond the end of file")
    {
        edit_chunk(fn, channel, 0, [](bytes &im, const h5::chunk &)
                   { REQUIRE(im[5] == 1); h5::put_le(&im[6], 65535, 2); });
        check_rejected();
    }
    SECTION("chunk 0 with a body > 65535 (4-byte size field)")
    {
        edit_chunk(fn, channel, 0, [](bytes &im, const h5::chunk &)
                   { im[5] = 2, h5::put_le(&im[6], 65536, 2); });
        check_rejected();
    }
    SECTION("message size beyond the body")
    {
        edit_chunk(fn, locate(fn, "channel_1/delay"), 0, [](bytes &im, const h5::chunk &c)
                   { h5::put_le(&im[message_in(c, h5::msg_datatype).offset - 3], 60000, 2); });
        check_rejected();
    }
    SECTION("unknown message type (0x11, symbol table) in a dataset")
    {
        edit_chunk(fn, locate(fn, "channel_1/delay"), 0, [](bytes &im, const h5::chunk &c)
                   { im[message_in(c, h5::msg_fill_value).offset - 4] = 0x11; });
        check_rejected();
    }
    SECTION("shared message flag")
    {
        edit_chunk(fn, locate(fn, "channel_1/delay"), 0, [](bytes &im, const h5::chunk &c)
                   { im[message_in(c, h5::msg_datatype).offset - 1] |= 0x02; });
        check_rejected();
    }

    // The 20 bytes reserved at the end of the group 'Data' take a Continuation message
    auto data_continuation = [&](uint64_t addr, uint64_t length)
    {
        edit_messages(fn, "channel_1/Data", 0, [&](std::vector<message> &m)
                      { m.push_back(continuation_message(addr, length)); });
    };
    SECTION("continuation beyond the end of file")
    {
        data_continuation(eof + 64, 64);
        check_data_rejected("");
    }
    SECTION("continuation backwards to its own chunk")
    {
        const uint64_t data = locate(fn, "channel_1/Data");
        data_continuation(data, object_at(fn, data).chunks[0].length());
        check_data_rejected("");
    }
    SECTION("continuation twice to the same chunk")
    {
        const uint64_t addr = eof;
        const bytes msgs = encode({continuation_message(addr, 28)}); // A chunk that continues with itself
        bytes image;
        h5::encode_chunk(image, false, msgs.data(), msgs.size(), 20);
        REQUIRE(append(fn, image) == addr);
        data_continuation(addr, 28);
        check_data_rejected("");
    }
    SECTION("continuation length 65544 and 5 GB")
    {
        data_continuation(48, 65544);
        check_data_rejected("");
        standard_file(fn);
        data_continuation(48, 5000000000ULL);
        check_data_rejected("");
    }
    SECTION("Link Info with a fractal heap (dense link storage)")
    {
        edit_messages(fn, "channel_1", 0, [](std::vector<message> &m)
                      { h5::put_le(&m[find(m, h5::msg_link_info)].data[2], 4096, 8); });
        check_rejected();
    }
    SECTION("Attribute Info with a fractal heap (dense attribute storage)")
    {
        edit_messages(fn, "channel_1", 0, [](std::vector<message> &m)
                      { h5::put_le(&m[find(m, h5::msg_attribute_info)].data[2], 4096, 8); });
        check_rejected();
    }
    SECTION("Attribute Info missing in a channel group")
    {
        edit_messages(fn, "channel_1", 0, [](std::vector<message> &m)
                      { m.erase(m.begin() + find(m, h5::msg_attribute_info)); });
        check_rejected();
    }
    SECTION("two Attribute Info messages")
    {
        edit_messages(fn, "channel_1", 0, [](std::vector<message> &m)
                      { m[find(m, h5::msg_attribute, "rx_orientation")] = attribute_info_message(); });
        check_rejected();
    }
    SECTION("Attribute Info in a dataset")
    {
        edit_messages(fn, "channel_1/delay", 0, [](std::vector<message> &m)
                      { m.push_back(attribute_info_message()); });
        check_rejected();
    }
    SECTION("chunked layout")
    {
        edit_messages(fn, "channel_1/coeff_re", 0, [](std::vector<message> &m)
                      { m[find(m, h5::msg_layout)].data[1] = 2; });
        check_rejected();
        standard_file(fn);
        edit_messages(fn, "channel_1/Data/par_big", 0, [](std::vector<message> &m)
                      { m[find(m, h5::msg_layout)].data[1] = 2; });
        check_data_rejected("big");
    }
    SECTION("datatype int16")
    {
        edit_messages(fn, "channel_1/Data/par_u", 0, [](std::vector<message> &m)
                      {
                          bytes &t = m[find(m, h5::msg_datatype)].data;
                          h5::put_le(&t[4], 2, 4);  // Size
                          h5::put_le(&t[10], 16, 2); }); // Precision
        check_data_rejected("u");
    }
    SECTION("datatype NUL-padded string")
    {
        edit_messages(fn, "channel_1/Data/par_s", 0, [](std::vector<message> &m)
                      { m[find(m, h5::msg_datatype)].data[1] = 0x01; });
        check_data_rejected("s");
    }
    SECTION("datatype float32 with another exponent bias")
    {
        edit_messages(fn, "channel_1/delay", 0, [](std::vector<message> &m)
                      { m[find(m, h5::msg_datatype)].data[16] = 0x7e; });
        check_rejected();
    }
    SECTION("dataspace with maximum extents, rank 4, null dataspace")
    {
        edit_messages(fn, "channel_1/delay", 0, [](std::vector<message> &m)
                      { m[find(m, h5::msg_dataspace)].data[2] = 1; });
        check_rejected();
        standard_file(fn);
        edit_messages(fn, "channel_1/delay", 0, [](std::vector<message> &m)
                      { m[find(m, h5::msg_dataspace)] = dataspace_message({1, 50, 2, 4}); });
        check_rejected();
        standard_file(fn);
        edit_messages(fn, "channel_1/delay", 0, [](std::vector<message> &m)
                      { m[find(m, h5::msg_dataspace)] = dataspace_message({}, 2); });
        check_rejected();
    }
}

TEST_CASE("hdf5 - channel data that contradicts the channel attributes is rejected")
{
    const std::string fn = "test_hdf_rb_semantics.hdf5";
    test_files files({fn});
    standard_file(fn);

    auto set_u32 = [&](const std::string &path, const char *name, unsigned value)
    {
        edit_messages(fn, path, 0, [&](std::vector<message> &m)
                      {
                          bytes &data = m[find(m, h5::msg_attribute, name)].data;
                          h5::attribute a;
                          REQUIRE(h5::decode_attribute(data.data(), data.size(), a));
                          h5::put_le(&data[a.value - data.data()], value, 4); });
    };
    auto set_dataspace = [&](const std::string &path, std::vector<uint64_t> dims)
    {
        edit_messages(fn, path, 0, [&](std::vector<message> &m)
                      { m[find(m, h5::msg_dataspace)] = dataspace_message(dims); });
    };
    auto remove_link = [&](const std::string &path, const char *name)
    {
        edit_messages(fn, path, 0, [&](std::vector<message> &m)
                      { m.erase(m.begin() + find(m, h5::msg_link, name)); });
    };
    auto rename_link = [&](const std::string &path, const char *name, const std::string &new_name)
    {
        edit_messages(fn, path, 0, [&](std::vector<message> &m)
                      {
                          message &l = m[find(m, h5::msg_link, name)];
                          l = link_message(new_name, h5::get_le(&l.data[l.data.size() - 8], 8)); });
    };
    auto check_rejected = [&](unsigned slot)
    {
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn, slot), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<float>(fn, slot), std::invalid_argument);
    };

    SECTION("NumPath 50 -> 1")
    {
        set_u32("channel_1", "NumPath", 1);
        check_rejected(0);
    }
    SECTION("NumRx 4 -> 1")
    {
        set_u32("channel_1", "NumRx", 1);
        check_rejected(0);
    }
    SECTION("link coeff_im removed")
    {
        remove_link("channel_1", "coeff_im");
        check_rejected(0);
    }
    SECTION("coeff_im without coeff_re")
    {
        remove_link("channel_1", "coeff_re");
        check_rejected(0);
    }
    SECTION("delay of rank 1 and of rank 4")
    {
        set_dataspace("channel_1/delay", {400});
        check_rejected(0);
        standard_file(fn);
        set_dataspace("channel_1/delay", {2, 25, 2, 4});
        check_rejected(0);
    }
    SECTION("Name as a 300-byte string")
    {
        edit_messages(fn, "channel_1", 0, [&](std::vector<message> &m)
                      {
                          bytes b;
                          h5::header hdr;
                          const std::string name(300, 'x');
                          hdr.attribute("Name", h5::string_type(300), 0, nullptr, name.data());
                          hdr.emit(b, 0, 0);
                          h5::chunk c;
                          c.image = b;
                          h5::parse_chunk(c);
                          m[find(m, h5::msg_attribute, "Name")] = messages_of(c)[1]; }); // After the Attribute Info
        check_rejected(0);
    }
    SECTION("NumPath 2^30 with matching extents but the original data (no 64 GB allocation)")
    {
        set_u32("channel_1", "NumPath", 1u << 30);
        for (const char *name : {"coeff_re", "coeff_im", "delay"})
            set_dataspace(std::string("channel_1/") + name, {1ULL << 30, 2, 4});
        check_rejected(0);
    }
    SECTION("NumSnap 4e9 with one snapshot group (no allocation for 4e9 snapshots)")
    {
        set_u32("channel_1", "NumSnap", 4000000000u);
        const uint64_t target = locate(fn, "channel_1");
        edit_messages(fn, "channel_1", 0, [&](std::vector<message> &m)
                      { m.push_back(link_message("Snap_0", target)); });
        check_rejected(0);
    }
    SECTION("Snap_1 missing in a 3-snapshot channel")
    {
        remove_link("channel_2", "Snap_1");
        check_rejected(1);
    }
    SECTION("snapshot group names with a leading zero, beyond 2^32 - 1, with 25 digits")
    {
        for (const std::string &name : {std::string("Snap_01"), std::string("Snap_4294967296"), "Snap_" + std::string(25, '1')})
        {
            INFO("name " << name);
            standard_file(fn);
            rename_link("channel_2", "Snap_1", name);
            check_rejected(1);
        }
    }
    SECTION("interact_coord with one row less than sum(no_interact)")
    {
        // Snapshot 0 has 3 paths with no_interact {0, 1, 2}: interact_coord [3, 3] is stored as {3, 3}
        edit_messages(fn, "channel_2/Snap_0/interact_coord", 0, [](std::vector<message> &m)
                      {
                          m[find(m, h5::msg_dataspace)] = dataspace_message({2, 3});
                          bytes &layout = m[find(m, h5::msg_layout)].data;
                          REQUIRE(h5::get_le(&layout[2], 2) == 36);
                          h5::put_le(&layout[2], 24, 2);
                          layout.resize(4 + 24); });
        check_rejected(1);
    }
    SECTION("compact unstructured data size other than elements times element size")
    {
        edit_messages(fn, "channel_1/Data/par_u", 0, [](std::vector<message> &m)
                      {
                          bytes &layout = m[find(m, h5::msg_layout)].data;
                          h5::put_le(&layout[2], 8, 2);
                          layout.resize(4 + 8); });
        check_rejected(0);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset(fn, "u", 0), std::invalid_argument);
    }
    SECTION("a link in 'Data' to a group")
    {
        redirect(fn, "channel_1/Data/par_x", locate(fn, "channel_1"));
        check_rejected(0);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset(fn, "x", 0), std::invalid_argument);
        CHECK(std::any_cast<std::string>(quadriga_lib::hdf5_read_dset(fn, "s", 0)) == "abc");
    }
    SECTION("unstructured dataset of rank 4")
    {
        set_dataspace("channel_1/Data/par_u", {1, 1, 1, 3});
        check_rejected(0);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset(fn, "u", 0), std::invalid_argument);
    }
    SECTION("the unpatched file reads back")
    {
        const auto c = quadriga_lib::hdf5_read_channel<double>(fn, 0);
        CHECK(c.is_valid().empty());
        CHECK(c.coeff_im[0].n_slices == 50);
        CHECK(arma::all(arma::vectorise(c.coeff_im[0]) == -0.5));
        CHECK(c.par_names == std::vector<std::string>({"x", "s", "u", "big"}));
        const auto c1 = quadriga_lib::hdf5_read_channel<double>(fn, 1);
        CHECK(c1.is_valid().empty());
        CHECK(arma::all(c1.n_path() == arma::uvec({3, 0, 2})));
    }
}

TEST_CASE("hdf5 - snapshot groups linked several times may not read more data than the file holds")
{
    const std::string fn = "test_hdf_rb_shared.hdf5";
    test_files files({fn});

    // Snapshot 0 holds most of the data; Snap_1 and Snap_2 are redirected to Snap_0, so the channel reads it 3 times
    auto build = [&](bool large_second_channel)
    {
        std::remove(fn.c_str());
        quadriga_lib::hdf5_create(fn, 2);
        auto c = full_channel({600, 1, 1});
        quadriga_lib::hdf5_write(&c, fn, 0);
        if (large_second_channel)
        {
            auto c1 = full_channel({2000});
            quadriga_lib::hdf5_write(&c1, fn, 1);
        }
        const uint64_t snap0 = locate(fn, "channel_1/Snap_0");
        redirect(fn, "channel_1/Snap_1", snap0);
        redirect(fn, "channel_1/Snap_2", snap0);
        QD_DIAG("\n# diag: shared snapshot groups\nsecond channel,end of file\n%d,%llu\n", (int)large_second_channel,
                (unsigned long long)eof_of(fn));
    };

    build(false);
    CHECK_THROWS_MATCHES(quadriga_lib::hdf5_read_channel<double>(fn, 0), std::invalid_argument,
                         Message("Malformed HDF5 file: the channel references more data than the file holds."));

    build(true); // The file is large enough for three reads of snapshot 0
    const auto c = quadriga_lib::hdf5_read_channel<double>(fn, 0);
    REQUIRE(c.n_snap() == 3);
    CHECK(arma::all(c.n_path() == arma::uvec({600, 600, 600})));
    for (size_t s = 1; s < 3; ++s)
    {
        CHECK(arma::approx_equal(c.coeff_re[s], c.coeff_re[0], "absdiff", 0.0));
        CHECK(arma::approx_equal(c.interact_coord[s], c.interact_coord[0], "absdiff", 0.0));
    }
}

TEST_CASE("hdf5 - snapshot links to one empty snapshot group may not read more headers than the file holds")
{
    // A channel group with 2000 links Snap_0 ... Snap_1999 to one snapshot group without paths: each link costs 23 file
    // bytes, but every read of the snapshot reads its 12 headers (about 1 KB) again
    const std::string fn = "test_hdf_rb_shared_headers.hdf5";
    test_files files({fn});
    quadriga_lib::hdf5_create(fn, 2);
    auto c = full_channel({0, 0});
    quadriga_lib::hdf5_write(&c, fn, 0);
    const uint64_t snap0 = locate(fn, "channel_1/Snap_0");

    auto add_channel = [&](uint32_t n_snap)
    {
        h5::header hdr;
        hdr.link_info(false);
        hdr.group_info();
        const uint32_t n_tx = 3, n_rx = 2; // As in the snapshot's coefficients
        const float pos[3] = {0.0f, 0.0f, 1.0f};
        const uint64_t three = 3;
        hdr.attribute("NumSnap", h5::uint32, 0, nullptr, &n_snap);
        hdr.attribute("NumTx", h5::uint32, 0, nullptr, &n_tx);
        hdr.attribute("NumRx", h5::uint32, 0, nullptr, &n_rx);
        hdr.attribute("tx_position", h5::float32, 1, &three, pos);
        hdr.attribute("rx_position", h5::float32, 1, &three, pos);
        for (uint32_t i = 0; i < n_snap; ++i)
            hdr.link("Snap_" + std::to_string(i), snap0);
        bytes image;
        hdr.emit(image, eof_of(fn), 0);
        const uint64_t addr = append(fn, image);
        bytes entry(8);
        h5::put_le(entry.data(), addr, 8);
        put(fn, contiguous_data(fn, "ChannelRef").first + 8, entry); // Slot 1
    };

    // Compact data is part of its header and counted once: a dataset of 4000 bytes in a file of about 7 KB reads back
    {
        const std::string fn_compact = "test_hdf_rb_shared_headers_compact.hdf5";
        test_files compact_files({fn_compact});
        quadriga_lib::hdf5_create(fn_compact, 1);
        std::any v = arma::regspace<arma::vec>(1.0, 500.0);
        quadriga_lib::hdf5_write_dset(fn_compact, "v", &v);
        CHECK(std::any_cast<arma::vec>(quadriga_lib::hdf5_read_dset(fn_compact, "v")).n_elem == 500);
        CHECK(quadriga_lib::hdf5_read_channel<double>(fn_compact).par_names.size() == 1);
    }

    add_channel(3); // Small: the shared snapshot reads back three times
    CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 1).n_snap() == 3);

    add_channel(2000);
    QD_DIAG("\n# diag: 2000 links to one empty snapshot group\nend of file\n%llu\n", (unsigned long long)eof_of(fn));
    CHECK_THROWS_MATCHES(quadriga_lib::hdf5_read_channel<double>(fn, 1), std::invalid_argument,
                         Message("Malformed HDF5 file: the channel references more data than the file holds."));
    CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 0).n_snap() == 2); // A valid channel of the same file
}

TEST_CASE("hdf5 - empty datasets with huge extents and huge interaction counts allocate nothing beyond the file")
{
    // Run with the allocation limit of the sanitizer build: an allocation sized by these extents fails the case
    const std::string fn = "test_hdf_rb_extents.hdf5";
    test_files files({fn});
    auto set_dataspace = [&](const std::string &path, std::vector<uint64_t> dims)
    {
        edit_messages(fn, path, 0, [&](std::vector<message> &m)
                      { m[find(m, h5::msg_dataspace)] = dataspace_message(dims); });
    };

    SECTION("unstructured cube with 2^29 empty slices and with extents whose product exceeds 64 bits")
    {
        quadriga_lib::hdf5_create(fn, 4);
        std::any x = arma::fcube(0, 0, 5);
        quadriga_lib::hdf5_write_dset(fn, "c", &x, 0);
        for (std::vector<uint64_t> dims : {std::vector<uint64_t>{1ULL << 29, 0, 0}, {0, 1ULL << 40, 1ULL << 40}})
        {
            set_dataspace("channel_1/Data/par_c", dims);
            CHECK_THROWS_AS(quadriga_lib::hdf5_read_dset(fn, "c", 0), std::invalid_argument);
            CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<float>(fn, 0), std::invalid_argument);
        }
    }
    SECTION("delay with 2^29 empty slices and NumPath 2^29")
    {
        quadriga_lib::hdf5_create(fn, 4);
        quadriga_lib::channel<double> c;
        c.tx_pos = arma::vec({0.0, 0.0, 1.0});
        c.rx_pos = arma::vec({10.0, 0.0, 1.5});
        c.delay = {arma::cube(1, 0, 3)};
        quadriga_lib::hdf5_write(&c, fn, 0, 0, 0, 0, true);
        set_dataspace("channel_1/delay", {1ULL << 29, 0, 1});
        edit_messages(fn, "channel_1", 0, [&](std::vector<message> &m)
                      {
                          bytes &data = m[find(m, h5::msg_attribute, "NumPath")].data;
                          h5::put_le(&data[data.size() - 4], 1ULL << 29, 4); });
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<float>(fn, 0), std::invalid_argument);
    }
    SECTION("interaction counts of 2^32 - 1 and 2^29 without matching coordinates")
    {
        quadriga_lib::channel<double> c = minimal_channel(1.0);
        c.coeff_re = {arma::cube(1, 1, 2, arma::fill::ones)};
        c.coeff_im = {arma::cube(1, 1, 2, arma::fill::zeros)};
        c.delay = {arma::cube(1, 1, 2, arma::fill::value(1e-6))};
        c.no_interact = {arma::u32_vec({1, 0})};
        c.interact_coord = {arma::mat(3, 1, arma::fill::ones)};
        for (std::pair<uint32_t, uint32_t> counts : {std::pair<uint32_t, uint32_t>{0xFFFFFFFFu, 0xFFFFFFFFu}, {1u << 29, 0u}})
        {
            quadriga_lib::hdf5_create(fn, 4);
            quadriga_lib::hdf5_write(&c, fn, 0);
            edit_messages(fn, "channel_1/no_interact", 0, [&](std::vector<message> &m)
                          {
                              bytes &layout = m[find(m, h5::msg_layout)].data; // 03 00, size, compact data
                              REQUIRE(h5::get_le(&layout[2], 2) == 8);
                              h5::put_le(&layout[4], counts.first, 4);
                              h5::put_le(&layout[8], counts.second, 4); });
            CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn, 0), std::invalid_argument);
        }
    }
}

TEST_CASE("hdf5 - index data that overlaps object headers is rejected before a writer changes the file")
{
    const std::string fn = "test_hdf_rb_overlap.hdf5";
    test_files files({fn});
    auto set_data_address = [&](const std::string &path, uint64_t addr)
    {
        edit_messages(fn, path, 0, [&](std::vector<message> &m)
                      { h5::put_le(&m[find(m, h5::msg_layout)].data[2], addr, 8); });
    };

    SECTION("root datasets: data inside root chunk 0 and inside another dataset header")
    {
        // Layout of a new file with 4 slots: root chunk 0 [48, 501), ChannelDims header 501, Version 588, Order
        // header 641 (data 712), ChannelRef header 728 (data 795)
        for (std::pair<const char *, uint64_t> p : {std::pair<const char *, uint64_t>{"ChannelRef", 400}, {"ChannelDims", 712 - 16},
                                                     {"Order", 724}, {"ChannelRef", 712}})
        {
            INFO("data of " << p.first << " at " << p.second);
            quadriga_lib::hdf5_create(fn, 4);
            set_data_address(p.first, p.second);
            check_all_throw(fn, all_functions(fn));
        }
    }
    SECTION("a root link chunk inside the index data")
    {
        quadriga_lib::hdf5_create(fn, 4);
        auto c = minimal_channel(1.0);
        quadriga_lib::hdf5_write(&c, fn, 0);
        const region chunk = link_chunk(fn, 0);
        set_data_address("Order", chunk.addr); // Order [chunk.addr, + 16): the slot IDs are bytes of the link chunk
        CHECK_NOTHROW(quadriga_lib::hdf5_read_channel<double>(fn, 0));
        const bytes before = load(fn);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write(&c, fn, 1), std::invalid_argument);
        std::any x = 1.0;
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(fn, "x", &x, 2), std::invalid_argument);
        CHECK(load(fn) == before);
    }
}

#if !defined(_WIN32)
TEST_CASE("hdf5 - a file that exists but cannot be read is not replaced")
{
    const std::string fn = "test_hdf_rb_write_only.hdf5";
    test_files files({fn});
    quadriga_lib::hdf5_create(fn, 4);
    auto c = minimal_channel(1.0);
    quadriga_lib::hdf5_write(&c, fn, 0);
    const bytes before = load(fn);
    namespace fs = std::filesystem;
    fs::permissions(fn, fs::perms::owner_write, fs::perm_options::replace);
    if (FILE *probe = std::fopen(fn.c_str(), "rb")) // E.g. running as root: the file stays readable
        std::fclose(probe);
    else
    {
        std::any x = 1.0;
        CHECK_THROWS_AS(quadriga_lib::hdf5_write(&c, fn, 1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_write_dset(fn, "x", &x, 1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_layout(fn), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_read_channel<double>(fn, 0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::hdf5_reshape_layout(fn, 4), std::invalid_argument);
    }
    fs::permissions(fn, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
    CHECK(load(fn) == before);
    CHECK(value_of(fn, 0) == 1.0);
}
#endif

TEST_CASE("hdf5 - writers reject damaged or foreign root and group structures before changing the file")
{
    // 48 slots: 0-44 minimal channels (IDs 1-45; root link chunk 0 holds IDs 1-41, link chunk 1 IDs 42-45), slot 1
    // extended with a dataset (new group 'Data'), slot 46 created by hdf5_write_dset (ID 46), slot 47 empty
    const std::string fn = "test_hdf_rb_writers.hdf5", fn_w = "test_hdf_rb_writers_copy.hdf5";
    test_files files({fn, fn_w});
    quadriga_lib::hdf5_create(fn, 48);
    for (unsigned s = 0; s < 45; ++s)
    {
        auto c = minimal_channel(s + 1.0);
        quadriga_lib::hdf5_write(&c, fn, s);
    }
    std::any a = 1.0;
    quadriga_lib::hdf5_write_dset(fn, "a", &a, 1);
    quadriga_lib::hdf5_write_dset(fn, "a", &a, 46);
    REQUIRE(link_chunk_ids(fn, 0).size() == 41);
    REQUIRE(link_chunk_ids(fn, 1) == std::vector<long long>({42, 43, 44, 45, 46}));

    // The writers, each with a check of its result
    struct writer
    {
        const char *name;
        std::function<void(const std::string &)> run, check;
    };
    std::any z = 9.0;
    const std::vector<writer> writers = {
        {"hdf5_write, empty slot", [](const std::string &f)
         { auto c = minimal_channel(100.0); quadriga_lib::hdf5_write(&c, f, 47); },
         [](const std::string &f)
         { CHECK(value_of(f, 47) == 100.0); }},
        {"hdf5_write, overwrite of slot 0 (link chunk 0)", [](const std::string &f)
         { auto c = minimal_channel(100.0); quadriga_lib::hdf5_write(&c, f, 0); },
         [](const std::string &f)
         { CHECK(value_of(f, 0) == 100.0); }},
        {"hdf5_write, overwrite of slot 44 (last link chunk)", [](const std::string &f)
         { auto c = minimal_channel(100.0); quadriga_lib::hdf5_write(&c, f, 44); },
         [](const std::string &f)
         { CHECK(value_of(f, 44) == 100.0); }},
        {"hdf5_write, overwrite of slot 1 (with 'Data')", [](const std::string &f)
         { auto c = minimal_channel(100.0); quadriga_lib::hdf5_write(&c, f, 1); },
         [](const std::string &f)
         { CHECK(value_of(f, 1) == 100.0); CHECK(dset_names(f, 1).empty()); }},
        {"hdf5_write_dset, empty slot", [&](const std::string &f)
         { quadriga_lib::hdf5_write_dset(f, "z", &z, 47); },
         [](const std::string &f)
         { CHECK(std::any_cast<double>(quadriga_lib::hdf5_read_dset(f, "z", 47)) == 9.0); }},
        {"hdf5_write_dset, slot 1 (append to 'Data')", [&](const std::string &f)
         { quadriga_lib::hdf5_write_dset(f, "z", &z, 1); },
         [](const std::string &f)
         { CHECK(dset_names(f, 1) == std::vector<std::string>({"a", "z"})); }},
        {"hdf5_write_dset, slot 2 (new 'Data')", [&](const std::string &f)
         { quadriga_lib::hdf5_write_dset(f, "z", &z, 2); },
         [](const std::string &f)
         { CHECK(dset_names(f, 2) == std::vector<std::string>({"z"})); CHECK(value_of(f, 2) == 3.0); }},
        {"hdf5_reshape_layout", [](const std::string &f)
         { quadriga_lib::hdf5_reshape_layout(f, 6, 8); },
         [](const std::string &f)
         { CHECK(arma::all(quadriga_lib::hdf5_read_layout(f) == arma::u32_vec({6, 8, 1, 1}))); }}};

    // Each writer on a fresh copy: the writers in 'rejected' throw (with 'msg' if not empty) and leave the file
    // unchanged, the others succeed and read back. The read functions are not affected.
    auto check_writers = [&](std::set<std::string> rejected, const std::string &msg = "")
    {
        const bytes patched = load(fn);
        for (const writer &w : writers)
        {
            INFO("writer " << w.name);
            save(fn_w, patched);
            if (rejected.count(w.name) != 0)
            {
                if (msg.empty())
                    CHECK_THROWS_AS(w.run(fn_w), std::invalid_argument);
                else
                    CHECK_THROWS_MATCHES(w.run(fn_w), std::invalid_argument, Message(msg));
                CHECK(load(fn_w) == patched);
            }
            else
            {
                CHECK_NOTHROW(w.run(fn_w));
                w.check(fn_w);
            }
        }
    };
    const std::set<std::string> root_writers = {"hdf5_write, empty slot", "hdf5_write, overwrite of slot 0 (link chunk 0)",
                                                "hdf5_write, overwrite of slot 44 (last link chunk)",
                                                "hdf5_write, overwrite of slot 1 (with 'Data')", "hdf5_write_dset, empty slot"};
    std::set<std::string> all;
    for (const writer &w : writers)
        all.insert(w.name);
    auto check_readers = [&]()
    {
        CHECK(value_of(fn, 0) == 1.0);
        CHECK(value_of(fn, 44) == 45.0);
        CHECK(dset_names(fn, 1) == std::vector<std::string>({"a"}));
    };
    const uint64_t root = locate(fn, "");

    SECTION("last root link chunk with a flipped byte")
    {
        edit_chunk(fn, root, 2, [](bytes &im, const h5::chunk &)
                   { im[40] ^= 0x01; }, false);
        check_writers(root_writers);
        check_readers();
    }
    SECTION("chunk of 'Data' with a flipped byte")
    {
        edit_chunk(fn, locate(fn, "channel_2/Data"), 0, [](bytes &im, const h5::chunk &)
                   { im[20] ^= 0x01; }, false);
        check_writers({"hdf5_write_dset, slot 1 (append to 'Data')"});
        CHECK_THROWS_AS(dset_names(fn, 1), std::invalid_argument);
    }
    SECTION("ChannelDims corrupted")
    {
        bytes b(16);
        h5::put_le(b.data(), 47, 4);
        put(fn, contiguous_data(fn, "ChannelDims").first, b);
        check_writers(all, corrupted_message);
    }
    SECTION("a foreign link in root chunk 0 after the directory")
    {
        const uint64_t target = locate(fn, "Order");
        edit_messages(fn, "", 0, [&](std::vector<message> &m)
                      { m.push_back(link_message("foreign", target)); });
        check_writers(all, unsupported_message);
        check_readers();
    }
    SECTION("root chunk 0 with free space that is not a multiple of 20")
    {
        shrink_chunk0(fn, "", 10);
        check_writers(all, unsupported_message);
        check_readers();
    }

    // Link chunk k (1 = last, 0 = holds the link of slot 0) with one change
    auto edit_link_chunk = [&](size_t k, const std::function<void(std::vector<message> &)> &edit)
    { edit_messages(fn, "", k + 1, edit); };
    const std::vector<std::pair<std::string, std::function<void(std::vector<message> &)>>> link_chunk_edits = {
        {"a Continuation message", [](std::vector<message> &m)
         { m[1] = continuation_message(48, 64); }},
        {"a link with a leading zero", [](std::vector<message> &m)
         { m[1] = link_message("channel_07", h5::get_le(&m[1].data[m[1].data.size() - 8], 8)); }},
        {"a link beyond 2^32 - 1", [](std::vector<message> &m)
         { m.back() = link_message("channel_4294967296", h5::get_le(&m.back().data[m.back().data.size() - 8], 8)); }},
        {"decreasing IDs", [](std::vector<message> &m)
         { std::swap(m[1], m[2]); }}};

    SECTION("last root link chunk outside the subset")
    {
        const bytes good = load(fn);
        for (const auto &e : link_chunk_edits)
        {
            INFO("edit " << e.first);
            save(fn, good);
            edit_link_chunk(1, e.second);
            check_writers(root_writers, unsupported_message);
            check_readers();
        }
        save(fn, good);
        edit_link_chunk(1, [](std::vector<message> &m)
                        { m.clear(); });
        check_writers(root_writers, unsupported_message);
    }
    SECTION("root link chunk of an overwritten slot outside the subset")
    {
        const bytes good = load(fn);
        for (const auto &e : link_chunk_edits)
        {
            INFO("edit " << e.first);
            save(fn, good);
            edit_link_chunk(0, e.second);
            check_writers({"hdf5_write, overwrite of slot 0 (link chunk 0)", "hdf5_write, overwrite of slot 1 (with 'Data')"},
                          unsupported_message);
            check_readers();
        }
    }
    SECTION("NIL messages with non-zero data are accepted")
    {
        for (const std::string path : {"", "channel_3"})
            edit_chunk(fn, locate(fn, path), 0, [](bytes &im, const h5::chunk &c)
                       {
                           const h5::message_ref &nil = c.messages.back();
                           REQUIRE(nil.type == h5::msg_nil);
                           std::fill(im.begin() + nil.offset, im.begin() + nil.offset + nil.size, 0xab); });
        check_writers({});
        check_readers();
    }
    SECTION("a channel group whose 'Data' link became a NIL message takes a new 'Data' link")
    {
        edit_chunk(fn, locate(fn, "channel_46"), 0, [](bytes &im, const h5::chunk &c)
                   { im[message_in(c, h5::msg_link, "Data").offset - 4] = h5::msg_nil; });
        quadriga_lib::hdf5_write_dset(fn, "z", &z, 46);
        CHECK(dset_names(fn, 46) == std::vector<std::string>({"z"}));
        CHECK(object_at(fn, locate(fn, "channel_46")).chunks[0].free_space == 0);
    }
    SECTION("a channel group with 4 bytes of free space cannot take the 'Data' link")
    {
        edit_messages(fn, "channel_46", 0, [](std::vector<message> &m)
                      { m.erase(m.begin() + find(m, h5::msg_link, "Data")); });
        shrink_chunk0(fn, "channel_46", 15);
        REQUIRE(object_at(fn, locate(fn, "channel_46")).chunks[0].free_space == 4);
        const bytes before = load(fn);
        CHECK_THROWS_MATCHES(quadriga_lib::hdf5_write_dset(fn, "z", &z, 46), std::invalid_argument, Message(unsupported_message));
        CHECK(load(fn) == before);
    }
    SECTION("a group 'Data' with 10 bytes of free space or without creation order cannot take a link")
    {
        const bytes good = load(fn);
        shrink_chunk0(fn, "channel_2/Data", 10);
        bytes before = load(fn);
        CHECK_THROWS_MATCHES(quadriga_lib::hdf5_write_dset(fn, "z", &z, 1), std::invalid_argument, Message(unsupported_message));
        CHECK(load(fn) == before);

        save(fn, good);
        edit_messages(fn, "channel_2/Data", 0, [](std::vector<message> &m)
                      {
                          bytes &info = m[find(m, h5::msg_link_info)].data;
                          info.erase(info.begin() + 2, info.begin() + 10);
                          info.resize(18), info[1] = 0; });
        before = load(fn);
        CHECK_THROWS_MATCHES(quadriga_lib::hdf5_write_dset(fn, "z", &z, 1), std::invalid_argument, Message(unsupported_message));
        CHECK(load(fn) == before);
    }
    SECTION("a chunk of 'Data' with two NIL messages is compacted when a link is added")
    {
        std::any b = 2.0, c = 3.0, d = 4.0;
        quadriga_lib::hdf5_write_dset(fn, "b", &b, 1); // New chunk
        quadriga_lib::hdf5_write_dset(fn, "c", &c, 1); // In the new chunk
        const uint64_t data = locate(fn, "channel_2/Data");
        REQUIRE(object_at(fn, data).chunks.size() == 2);
        edit_chunk(fn, data, 1, [](bytes &im, const h5::chunk &ch)
                   {
                       // Links b, c, NIL -> b, NIL (4 bytes), c, NIL (4 bytes less)
                       REQUIRE(ch.messages.size() == 3);
                       const h5::message_ref &mc = ch.messages[1], &nil = ch.messages[2];
                       REQUIRE(nil.size >= 4);
                       bytes msg_c(im.begin() + (mc.offset - 4), im.begin() + (mc.offset + mc.size));
                       std::copy(msg_c.begin(), msg_c.end(), im.begin() + mc.offset);
                       h5::put_le(&im[mc.offset - 4], 0, 4); // NIL with no data
                       h5::put_le(&im[nil.offset + 4 - 3], nil.size - 4, 2);
                       im[nil.offset + 4 - 4] = h5::msg_nil, im[nil.offset + 4 - 1] = 0; });
        REQUIRE(object_at(fn, data).chunks[1].messages.size() == 4);
        quadriga_lib::hdf5_write_dset(fn, "d", &d, 1);
        CHECK(dset_names(fn, 1) == std::vector<std::string>({"a", "b", "c", "d"}));
        CHECK(std::any_cast<double>(quadriga_lib::hdf5_read_dset(fn, "c", 1)) == 3.0);
        const h5::object o = object_at(fn, data);
        std::vector<std::string> names;
        for (const message &m : messages_of(o.chunks[1]))
            names.push_back(name_of(m));
        CHECK(names == std::vector<std::string>({"par_b", "par_c", "par_d"}));
        CHECK(o.chunks[1].messages.back().type == h5::msg_nil);
    }
}

TEST_CASE("hdf5 - single-byte flips in headers and index data give std::invalid_argument or a valid result")
{
    // Slot 0: written, overwritten by a 3-snapshot channel with every field and 3 unstructured datasets, then extended
    // by hdf5_write_dset (the group 'Data' grows a second chunk); slot 1: created by hdf5_write_dset
    const std::string fn = "test_hdf_rb_flip.hdf5", fn_v = "test_hdf_rb_flip_variant.hdf5",
                      fn_w = "test_hdf_rb_flip_writer.hdf5";
    test_files files({fn, fn_v, fn_w});
    quadriga_lib::hdf5_create(fn, 2);
    auto first = minimal_channel(1.0);
    quadriga_lib::hdf5_write(&first, fn, 0);
    auto c = full_channel({3, 0, 2});
    c.par_names = {"x", "s", "m"};
    c.par_data = {1.5, std::string("text"), arma::Mat<int>(2, 3, arma::fill::value(-4))};
    quadriga_lib::hdf5_write(&c, fn, 0);
    std::any e = 2.5, a = 3.5, w = 7.0;
    quadriga_lib::hdf5_write_dset(fn, "e", &e, 0);
    quadriga_lib::hdf5_write_dset(fn, "a", &a, 1);
    const bytes base = load(fn);

    // Every header chunk reachable from the root (each object once), and the superblock, which has its checksum in
    // the last 4 bytes as well. 'unread': chunks that no call reads (the header of an 'interact_coord' without
    // coordinates, which hdf5_read_channel skips).
    std::vector<region> headers = {{0, 48}};
    std::set<uint64_t> unread;
    {
        h5::file f;
        REQUIRE(f.open(fn, "rb"));
        const h5::superblock sb = h5::read_superblock(f);
        std::set<uint64_t> seen;
        std::vector<std::pair<uint64_t, std::string>> todo = {{sb.root, ""}};
        while (!todo.empty())
        {
            const auto [addr, name] = todo.back();
            todo.pop_back();
            if (!seen.insert(addr).second)
                continue;
            const h5::object o = h5::read_object(f, addr, sb.eof);
            bool is_group = false;
            for (const h5::chunk &ch : o.chunks)
            {
                headers.push_back({ch.addr, ch.length()});
                for (const h5::message_ref &m : ch.messages)
                    is_group |= m.type == h5::msg_link_info;
            }
            if (is_group)
            {
                const h5::group g = h5::decode_group(o);
                for (const h5::link &l : g.links)
                    todo.push_back({l.target, std::string(l.name)});
            }
            else if (name == "interact_coord" && h5::decode_dataset(o, sb.eof).dims[0] == 0)
                unread.insert(addr);
        }
    }
    REQUIRE(unread.size() == 1); // Snapshot 1 has no paths
    std::vector<region> index;
    for (const char *name : {"ChannelDims", "Order", "ChannelRef"})
    {
        const auto d = contiguous_data(fn, name);
        index.push_back({d.first, d.second * (std::string(name) == "ChannelRef" ? 8 : 4)});
    }

    enum outcome
    {
        ok,
        rejected,
        other
    };
    auto run = [](const std::function<void()> &f)
    {
        try
        {
            f();
            return ok;
        }
        catch (const std::invalid_argument &)
        {
            return rejected;
        }
        catch (...)
        {
            return other;
        }
    };

    // Read calls: outcome ok or rejected
    arma::u32_vec ids;
    std::vector<std::string> names;
    const std::vector<std::function<void()>> reads = {
        [&]()
        { quadriga_lib::hdf5_read_layout(fn_v, &ids); },
        [&]()
        { quadriga_lib::hdf5_read_channel<double>(fn_v, 0); },
        [&]()
        { quadriga_lib::hdf5_read_dset(fn_v, "a", 1); },
        [&]()
        { names.clear(), quadriga_lib::hdf5_read_dset_names(fn_v, &names, 0); }};
    // Writer calls on a copy: they succeed and read back, or throw std::invalid_argument and leave the file unchanged
    auto overwrite = minimal_channel(42.0);
    const std::vector<std::pair<std::function<void()>, std::function<bool()>>> writes = {
        {[&]()
         { quadriga_lib::hdf5_write_dset(fn_w, "flip", &w, 0); },
         [&]()
         { return std::any_cast<double>(quadriga_lib::hdf5_read_dset(fn_w, "flip", 0)) == 7.0; }},
        {[&]()
         { quadriga_lib::hdf5_write(&overwrite, fn_w, 0); },
         [&]()
         { return value_of(fn_w, 0) == 42.0; }}};

    struct counts
    {
        size_t variants = 0, read_ok = 0, read_rejected = 0, write_ok = 0, write_rejected = 0;
        std::vector<std::string> violations;
    };
    // Strides, fixed so that the positions are deterministic and every chunk is covered: the read calls run on every
    // variant of (a) and (c) and on every 2nd variant of (b), the writer calls on every 16th variant of (a) and (b),
    // on every variant of (a) that all read calls accept, and on every variant of (c); about 2 s in the release build.
    // The read calls use a copy of the base file that is changed in place for each variant and restored afterwards,
    // the writer calls a copy that is rewritten each time.
    save(fn_v, base);
    h5::file variant_file, variant_copy;
    REQUIRE(variant_file.open(fn_v, "r+b"));
    save(fn_w, base);
    REQUIRE(variant_copy.open(fn_w, "r+b"));
    // 'must_detect': at least one call must throw (a flip without a new checksum); if all read calls succeed, the
    // writer calls run as well
    auto check_variant = [&](const bytes &image, region changed, bool do_reads, bool do_writes, bool must_detect,
                             counts &n, const std::string &what)
    {
        ++n.variants;
        const size_t rejected_before = n.read_rejected + n.write_rejected;
        if (do_reads)
        {
            variant_file.write(changed.addr, &image[changed.addr], changed.size);
            for (size_t i = 0; i < reads.size(); ++i)
            {
                const outcome o = run(reads[i]);
                if (o == ok)
                    ++n.read_ok;
                else if (o == rejected)
                    ++n.read_rejected;
                else
                    n.violations.push_back(what + ": read " + std::to_string(i) + " threw another exception");
            }
            variant_file.write(changed.addr, &base[changed.addr], changed.size);
        }
        if (do_writes || (must_detect && n.read_rejected + n.write_rejected == rejected_before))
            for (size_t i = 0; i < writes.size(); ++i)
            {
                variant_copy.write(0, image.data(), image.size()); // Then cut to the size of the image
                std::filesystem::resize_file(fn_w, image.size());
                const outcome o = run(writes[i].first);
                if (o == ok)
                {
                    ++n.write_ok;
                    bool back = false;
                    try
                    {
                        back = writes[i].second();
                    }
                    catch (const std::exception &ex)
                    {
                        n.violations.push_back(what + ": write " + std::to_string(i) + " does not read back: " + ex.what());
                        continue;
                    }
                    if (!back)
                        n.violations.push_back(what + ": write " + std::to_string(i) + " reads back other values");
                }
                else if (o == rejected)
                {
                    ++n.write_rejected;
                    if (load(fn_w) != image)
                        n.violations.push_back(what + ": write " + std::to_string(i) + " threw after changing the file");
                }
                else
                    n.violations.push_back(what + ": write " + std::to_string(i) + " threw another exception");
            }
        if (must_detect && n.read_rejected + n.write_rejected == rejected_before)
            n.violations.push_back(what + ": not detected");
    };
    auto start = std::chrono::steady_clock::now();
    auto report = [&](const char *part, const counts &n)
    {
        const auto now = std::chrono::steady_clock::now();
        QD_DIAG("%s,%zu,%zu,%zu,%zu,%zu,%zu,%.2f\n", part, n.variants, n.read_ok, n.read_rejected, n.write_ok,
                n.write_rejected, n.violations.size(), std::chrono::duration<double>(now - start).count());
        start = now;
        INFO("part " << part);
        for (size_t i = 0; i < std::min(n.violations.size(), (size_t)20); ++i)
            UNSCOPED_INFO(n.violations[i]);
        CHECK(n.violations.empty());
    };

    size_t header_bytes = 0;
    for (const region &r : headers)
        header_bytes += r.size;
    QD_DIAG("\n# diag: byte-flip sweep, %zu header chunks with %zu bytes, file %zu bytes\n", headers.size(), header_bytes, base.size());
    QD_DIAG("part,variants,read ok,read rejected,write ok,write rejected,violations,seconds\n");

    // (a) Flips without a new checksum: the checksum verification; every flip must be detected
    counts na;
    size_t variant = 0;
    for (const region &r : headers)
        for (uint64_t p = r.addr; p < r.addr + r.size; ++p, ++variant)
        {
            bytes image = base;
            image[p] ^= 0xff;
            check_variant(image, {p, 1}, true, variant % 16 == 0, unread.count(r.addr) == 0, na, "(a) " + std::to_string(p));
        }
    report("(a) flip", na);

    // (b) Flips with a new checksum: the parsers behind the checksums
    counts nb;
    for (const uint8_t mask : {0xff, 0x01})
        for (const region &r : headers)
            for (uint64_t p = r.addr; p < r.addr + r.size - 4; ++p, ++variant)
            {
                bytes image = base;
                image[p] ^= mask;
                bytes chunk(image.begin() + r.addr, image.begin() + r.addr + r.size);
                resign(chunk);
                std::copy(chunk.begin(), chunk.end(), image.begin() + r.addr);
                check_variant(image, r, variant % 2 == 0, variant % 16 == 0, false, nb,
                              "(b) " + std::to_string(p) + " ^ " + std::to_string(mask));
            }
    report("(b) flip + checksum", nb);

    // (c) The data of ChannelDims, Order and ChannelRef (not checksummed)
    counts nc;
    for (const region &r : index)
        for (uint64_t p = r.addr; p < r.addr + r.size; ++p, ++variant)
        {
            bytes image = base;
            image[p] ^= 0xff;
            check_variant(image, {p, 1}, true, true, false, nc, "(c) " + std::to_string(p));
        }
    report("(c) index data", nc);
}

TEST_CASE("hdf5 - every intermediate state of an interrupted write is a valid file")
{
    // File A: 51 slots, slots 0-49 hold minimal channels with the values 1-50 (IDs 1-50; root link chunk 0 holds
    // IDs 1-41, link chunk 1 IDs 42-50), slot 50 is empty
    const std::string fn_a = "test_hdf_rb_int_a.hdf5", fn_b = "test_hdf_rb_int_b.hdf5", fn = "test_hdf_rb_int.hdf5";
    test_files files({fn_a, fn_b, fn});
    quadriga_lib::hdf5_create(fn_a, 51);
    for (unsigned s = 0; s < 50; ++s)
    {
        auto c = minimal_channel(s + 1.0);
        quadriga_lib::hdf5_write(&c, fn_a, s);
    }
    REQUIRE(link_chunk_ids(fn_a, 0).size() == 41);
    REQUIRE(link_chunk_ids(fn_a, 1).size() == 9);

    const auto ref = contiguous_data(fn_a, "ChannelRef"), order = contiguous_data(fn_a, "Order");
    std::any x = 7.0, y = 8.0;

    // The continuation of every state: an hdf5_write to an empty and to a filled slot succeed and read back with
    // the next IDs ('last_id' = ID of the last root link of the state)
    auto continue_writing = [&](unsigned last_id)
    {
        auto c = minimal_channel(200.0);
        CHECK(quadriga_lib::hdf5_write(&c, fn, 50) == 0);
        c = minimal_channel(300.0);
        CHECK(quadriga_lib::hdf5_write(&c, fn, 3) == 1);
        arma::u32_vec id;
        quadriga_lib::hdf5_read_layout(fn, &id);
        CHECK(id(50) == last_id + 1);
        CHECK(id(3) == last_id + 2);
        CHECK(value_of(fn, 50) == 200.0);
        CHECK(value_of(fn, 3) == 300.0);
        for (unsigned s : {1u, 41u, 49u})
            CHECK(value_of(fn, s) == s + 1.0);
    };
    // Regions of A and B: the appended block, the superblock, root link chunk k, the entries of a slot
    auto block = [&](const bytes &A, const bytes &B)
    { return region{A.size(), B.size() - A.size()}; };
    const region superblock = {0, 48};
    auto ref_entry = [&](unsigned slot)
    { return region{ref.first + 8 * slot, 8}; };
    auto order_entry = [&](unsigned slot)
    { return region{order.first + 4 * slot, 4}; };

    SECTION("hdf5_write overwrite: block, superblock, new link, ChannelRef, Order, removal of the old link")
    {
        save(fn_b, load(fn_a));
        auto c = minimal_channel(100.0);
        quadriga_lib::hdf5_write(&c, fn_b, 0);
        // Premise: the old link is in link chunk 0, the new one is appended to link chunk 1
        CHECK(link_chunk_ids(fn_a, 0)[0] == 1);
        CHECK(link_chunk_ids(fn_b, 0)[0] == 2);
        CHECK(link_chunk_ids(fn_b, 1).back() == 51);
        const bytes A = load(fn_a), B = load(fn_b);
        const auto states = interrupted_states(A, B, {block(A, B), superblock, link_chunk(fn_b, 1), ref_entry(0), order_entry(0), link_chunk(fn_b, 0)});

        // State i: the library reads the new channel from step 4a (ChannelRef) and its ID from step 4b (Order)
        for (size_t i = 0; i < states.size(); ++i)
        {
            INFO("state after step " << i + 1);
            save(fn, states[i]);
            arma::u32_vec id;
            quadriga_lib::hdf5_read_layout(fn, &id);
            CHECK(id(0) == (i >= 4 ? 51u : 1u));
            CHECK(value_of(fn, 0) == (i >= 3 ? 100.0 : 1.0));
            for (unsigned s = 1; s < 50; ++s)
                CHECK(value_of(fn, s) == s + 1.0);
            CHECK(value_of(fn, 50) == -1.0);
            continue_writing(i >= 2 ? 51 : 50);
        }
    }
    SECTION("hdf5_write overwrite with the old and the new link in the same link chunk")
    {
        quadriga_lib::hdf5_create(fn_a, 3);
        for (unsigned s = 0; s < 2; ++s)
        {
            auto c = minimal_channel(s + 1.0);
            quadriga_lib::hdf5_write(&c, fn_a, s);
        }
        save(fn_b, load(fn_a));
        auto c = minimal_channel(100.0);
        quadriga_lib::hdf5_write(&c, fn_b, 0);
        CHECK(link_chunk_ids(fn_b, 0) == std::vector<long long>({2, 3}));
        const bytes A = load(fn_a), B = load(fn_b);
        const auto r = contiguous_data(fn_b, "ChannelRef"), o = contiguous_data(fn_b, "Order");
        interrupted_states(A, B, {block(A, B), superblock, link_chunk(fn_b, 0), {r.first, 8}, {o.first, 4}});
        arma::u32_vec id;
        quadriga_lib::hdf5_read_layout(fn_b, &id);
        CHECK(arma::all(id == arma::u32_vec({3, 2, 0})));
        CHECK(value_of(fn_b, 0) == 100.0);
        CHECK(value_of(fn_b, 1) == 2.0);
    }
    SECTION("hdf5_write_dset on an empty slot: block, superblock, new link, ChannelRef, Order")
    {
        save(fn_b, load(fn_a));
        quadriga_lib::hdf5_write_dset(fn_b, "x", &x, 50);
        const bytes A = load(fn_a), B = load(fn_b);
        const auto states = interrupted_states(A, B, {block(A, B), superblock, link_chunk(fn_b, 1), ref_entry(50), order_entry(50)});
        for (size_t i = 0; i < states.size(); ++i)
        {
            INFO("state after step " << i + 1);
            save(fn, states[i]);
            arma::u32_vec id;
            quadriga_lib::hdf5_read_layout(fn, &id);
            CHECK(id(50) == (i >= 4 ? 51u : 0u)); // Stale after step 4a
            const std::any v = quadriga_lib::hdf5_read_dset(fn, "x", 50);
            CHECK(v.has_value() == (i >= 3));
            if (i >= 3)
            {
                CHECK(std::any_cast<double>(v) == 7.0);
                quadriga_lib::hdf5_write_dset(fn, "y", &y, 50); // Filled slot: Order is not written
                CHECK(dset_names(fn, 50) == std::vector<std::string>({"x", "y"}));
                quadriga_lib::hdf5_read_layout(fn, &id);
                CHECK(id(50) == (i >= 4 ? 51u : 0u));
                auto c = minimal_channel(300.0);
                CHECK(quadriga_lib::hdf5_write(&c, fn, 3) == 1);
                quadriga_lib::hdf5_read_layout(fn, &id);
                CHECK(id(3) == 52);
            }
            else
                continue_writing(i >= 2 ? 51 : 50);
        }
    }
    SECTION("hdf5_write_dset on a filled slot without 'Data': block, superblock, channel group")
    {
        save(fn_b, load(fn_a));
        quadriga_lib::hdf5_write_dset(fn_b, "x", &x, 1);
        const bytes A = load(fn_a), B = load(fn_b);
        const h5::object g = object_at(fn_b, locate(fn_b, "channel_2"));
        const auto states = interrupted_states(A, B, {block(A, B), superblock, {g.chunks[0].addr, g.chunks[0].length()}});
        for (size_t i = 0; i < states.size(); ++i)
        {
            INFO("state after step " << i + 1);
            save(fn, states[i]);
            CHECK(dset_names(fn, 1).size() == (i >= 2 ? 1u : 0u));
            CHECK(value_of(fn, 1) == 2.0);
            quadriga_lib::hdf5_write_dset(fn, "y", &y, 1);
            CHECK(dset_names(fn, 1) == (i >= 2 ? std::vector<std::string>({"x", "y"}) : std::vector<std::string>({"y"})));
            continue_writing(50);
        }
    }
    SECTION("hdf5_write_dset append: block, superblock, creation index in chunk 0, link in the last chunk")
    {
        quadriga_lib::hdf5_write_dset(fn_a, "a", &x, 2);
        quadriga_lib::hdf5_write_dset(fn_a, "b", &x, 2); // 'Data' grows a second chunk
        save(fn_b, load(fn_a));
        quadriga_lib::hdf5_write_dset(fn_b, "c", &y, 2);
        const bytes A = load(fn_a), B = load(fn_b);
        const h5::object d = object_at(fn_b, locate(fn_b, "channel_3/Data"));
        REQUIRE(d.chunks.size() == 2);
        const auto states = interrupted_states(A, B, {block(A, B), superblock, {d.chunks[0].addr, d.chunks[0].length()}, {d.chunks[1].addr, d.chunks[1].length()}});
        for (size_t i = 0; i < states.size(); ++i)
        {
            INFO("state after step " << i + 1);
            save(fn, states[i]);
            const std::vector<std::string> before = i >= 3 ? std::vector<std::string>({"a", "b", "c"}) : std::vector<std::string>({"a", "b"});
            CHECK(dset_names(fn, 2) == before);
            quadriga_lib::hdf5_write_dset(fn, "d", &y, 2); // After step 3a, the creation index has a gap
            std::vector<std::string> after = before;
            after.push_back("d");
            CHECK(dset_names(fn, 2) == after);
            CHECK(quadriga_lib::hdf5_read_channel<double>(fn, 2).par_names == after);
            continue_writing(50);
        }
    }
}

TEST_CASE("hdf5 - 8000 channels fill the root link directory and keep their IDs")
{
    const std::string fn = "test_hdf_rb_8000.hdf5";
    test_files files({fn});
    const unsigned n = 8000;
    quadriga_lib::hdf5_create(fn, n);
    for (unsigned s = 0; s < n; ++s)
    {
        auto c = minimal_channel(s + 1.0);
        quadriga_lib::hdf5_write(&c, fn, s);
    }
    // Overwrites: new IDs 8001-8005; the last-written slot twice more
    arma::u32_vec expected_id = arma::regspace<arma::u32_vec>(1, n);
    arma::vec expected_value = arma::regspace<arma::vec>(1.0, (double)n);
    unsigned next_id = n + 1;
    for (unsigned s : {0u, 3999u, 7999u, 7999u, 7999u})
    {
        auto c = minimal_channel(100000.0 + next_id);
        CHECK(quadriga_lib::hdf5_write(&c, fn, s) == 1);
        expected_id(s) = next_id, expected_value(s) = 100000.0 + next_id;
        ++next_id;
    }

    // Premise: the root links continue from chunk 0 into a directory chunk
    const h5::object root = object_at(fn, locate(fn, ""));
    size_t n_directory = 0, n_links = 0;
    for (size_t k = 1; k < root.chunks.size(); ++k)
    {
        const auto m = messages_of(root.chunks[k]);
        n_directory += !m.empty() && m[0].type == h5::msg_continuation;
        for (const auto &msg : m)
            n_links += msg.type == h5::msg_link;
    }
    QD_DIAG("\n# diag: 8000 channels\nroot chunks,directory chunks,links\n%zu,%zu,%zu\n", root.chunks.size(), n_directory, n_links);
    CHECK(n_directory >= 1);
    CHECK(n_links == n);

    arma::u32_vec id;
    quadriga_lib::hdf5_read_layout(fn, &id);
    CHECK(arma::all(id == expected_id));
    size_t wrong = 0;
    for (unsigned s = 0; s < n; ++s)
        wrong += value_of(fn, s) != expected_value(s);
    CHECK(wrong == 0);
}
