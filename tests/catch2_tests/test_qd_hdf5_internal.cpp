// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Focused tests of the private header qd_hdf5.hpp (the HDF5 subset used by the channel files): checksum, datatype
// and message encodings, chunk encoding with the tail rule, header splitting, the growable group headers, the root
// link directory, and bounds-checked file access. The channel file functions built on it are tested through the
// public API in test_hdf_functions.cpp; files written by the library are checked with the HDF5 C library in
// test_hdf5_interop.cpp.
//
// Oracles: published lookup3 test vectors and a superblock written by the HDF5 library 1.14.6; the byte encodings of
// the predefined datatypes as written by the HDF5 library; sizes and layouts derived by hand from the HDF5 File Format
// Specification Version 3.0 ("Version 2 Object Header", "Header Message Types").
// Fixtures: files are created with quadriga_lib::hdf5_create where a valid file is needed; each file is removed at
// the end of its test case.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_hdf5.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Diagnostic output: off by default. QD_DIAG_HDF=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way.
#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_HDF")

namespace h5 = qd::hdf5;

namespace
{
    // Removes the file when the test case starts and when it ends
    struct test_file
    {
        const std::string fn;
        explicit test_file(std::string name) : fn(std::move(name)) { std::remove(fn.c_str()); }
        ~test_file() { std::remove(fn.c_str()); }
    };

    std::vector<uint8_t> bytes(std::initializer_list<int> b)
    {
        std::vector<uint8_t> v;
        for (int x : b)
            v.push_back((uint8_t)x);
        return v;
    }

    // Messages of a chunk as their types
    std::vector<int> types(const h5::chunk &c)
    {
        std::vector<int> t;
        for (const auto &m : c.messages)
            t.push_back(m.type);
        return t;
    }

    // Appends the block of 'out' to the file and sets the end-of-file address in the superblock
    void commit(h5::file &f, h5::output &out, uint64_t root, const std::vector<h5::patch> &patches = {})
    {
        out.flush();
        const std::vector<uint8_t> sb = h5::superblock_bytes(out.addr(), root);
        f.write(0, sb.data(), sb.size());
        for (const auto &p : patches)
            f.write(p.addr, p.bytes.data(), p.bytes.size());
    }
}

TEST_CASE("qd_hdf5 - lookup3 matches the published test vectors")
{
    CHECK(h5::lookup3(nullptr, 0) == 0xdeadbeefu);
    const char *text = "Four score and seven years ago";
    CHECK(h5::lookup3((const uint8_t *)text, 30) == 0x17770551u);

    // Superblock written by the HDF5 library 1.14.6 (EOF 0xf9d, root 0x30); its stored checksum is 4e 2a d5 60
    std::vector<uint8_t> sb = bytes({0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a, 0x02, 0x08, 0x08, 0x00});
    sb.resize(20, 0x00);
    sb.resize(28, 0xff);
    for (int b : {0x9d, 0x0f, 0, 0, 0, 0, 0, 0, 0x30, 0, 0, 0, 0, 0, 0, 0})
        sb.push_back((uint8_t)b);
    REQUIRE(sb.size() == 44);
    CHECK(h5::lookup3(sb.data(), 44) == 0x60d52a4eu);
    CHECK(h5::superblock_bytes(0xf9d, 0x30) == [&]
          { auto s = sb; h5::put_le(s, 0x60d52a4eu, 4); return s; }());
}

TEST_CASE("qd_hdf5 - datatype messages equal the encodings of the HDF5 library")
{
    const std::vector<std::pair<h5::dtype, std::vector<uint8_t>>> table = {
        {h5::int32, bytes({0x10, 0x08, 0, 0, 4, 0, 0, 0, 0, 0, 0x20, 0})},
        {h5::uint32, bytes({0x10, 0x00, 0, 0, 4, 0, 0, 0, 0, 0, 0x20, 0})},
        {h5::int64, bytes({0x10, 0x08, 0, 0, 8, 0, 0, 0, 0, 0, 0x40, 0})},
        {h5::uint64, bytes({0x10, 0x00, 0, 0, 8, 0, 0, 0, 0, 0, 0x40, 0})},
        {h5::float32, bytes({0x11, 0x20, 0x1f, 0, 4, 0, 0, 0, 0, 0, 0x20, 0, 0x17, 0x08, 0, 0x17, 0x7f, 0, 0, 0})},
        {h5::float64, bytes({0x11, 0x20, 0x3f, 0, 8, 0, 0, 0, 0, 0, 0x40, 0, 0x34, 0x0b, 0, 0x34, 0xff, 0x03, 0, 0})},
        {h5::string_type(256), bytes({0x13, 0, 0, 0, 0, 1, 0, 0})},
        {h5::object_reference, bytes({0x17, 0, 0, 0, 8, 0, 0, 0})}};

    for (const auto &e : table)
    {
        INFO("type " << (int)e.first.t);
        std::vector<uint8_t> enc;
        h5::encode_datatype(enc, e.first);
        CHECK(enc == e.second);
        CHECK(h5::datatype_size(e.first) == e.second.size());
        h5::dtype t{};
        CHECK(h5::decode_datatype(e.second.data(), e.second.size(), t));
        CHECK(t == e.first);

        // A changed byte never decodes to the same type (string sizes are part of the type)
        const size_t n_check = e.first.t == h5::type_t::str ? 4 : e.second.size();
        for (size_t i = 0; i < n_check; ++i)
        {
            auto changed = e.second;
            changed[i] ^= 0x01;
            h5::dtype u{};
            const bool ok = h5::decode_datatype(changed.data(), changed.size(), u);
            INFO("byte " << i);
            CHECK((!ok || !(u == e.first)));
        }
    }

    // Other types are rejected: int16, NUL-padded string, float with another exponent bias, string of size 0
    h5::dtype t{};
    CHECK(!h5::decode_datatype(bytes({0x10, 0x08, 0, 0, 2, 0, 0, 0, 0, 0, 0x10, 0}).data(), 12, t));
    CHECK(!h5::decode_datatype(bytes({0x13, 0x01, 0, 0, 4, 0, 0, 0}).data(), 8, t));
    CHECK(!h5::decode_datatype(bytes({0x11, 0x20, 0x1f, 0, 4, 0, 0, 0, 0, 0, 0x20, 0, 0x17, 0x08, 0, 0x17, 0x7e, 0, 0, 0}).data(), 20, t));
    CHECK(!h5::decode_datatype(bytes({0x13, 0, 0, 0, 0, 0, 0, 0}).data(), 8, t));
}

TEST_CASE("qd_hdf5 - headers carry the Attribute Info message and the message flags of the HDF5 library")
{
    // Group with attributes: Link Info, Group Info, Attribute Info, attributes, links
    h5::header hdr;
    const uint32_t value = 7;
    const uint64_t three = 3;
    const float position[3] = {1.0f, 2.0f, 3.0f};
    hdr.link_info(false);
    hdr.group_info();
    hdr.attribute("NumSnap", h5::uint32, 0, nullptr, &value);
    hdr.attribute("tx_position", h5::float32, 1, &three, position);
    hdr.link("Data", 1000);
    h5::chunk c;
    hdr.emit(c.image, 2000, 0);
    c.addr = 2000;
    h5::parse_chunk(c);
    CHECK(types(c) == std::vector<int>({0x02, 0x0A, 0x15, 0x0C, 0x0C, 0x06}));
    REQUIRE(c.messages.size() == 6);
    const auto &ai = c.messages[2];
    std::vector<uint8_t> ai_bytes(c.image.begin() + (ai.offset - 4), c.image.begin() + (ai.offset + ai.size));
    std::vector<uint8_t> expected = bytes({0x15, 0x12, 0x00, 0x04, 0x00, 0x00});
    expected.resize(22, 0xff);
    CHECK(ai_bytes == expected);
    CHECK(c.messages[0].flags == 0x00);
    CHECK(c.messages[1].flags == 0x01);
    CHECK(c.messages[3].flags == 0x00);
    CHECK(c.messages[5].flags == 0x00);

    // The decoded view gives back names, types and values
    h5::object o;
    o.chunks.push_back(c);
    const h5::group g = h5::decode_group(o);
    REQUIRE(g.attributes.size() == 2);
    CHECK(g.attributes[1].name == "tx_position");
    CHECK(g.attributes[1].type == h5::float32);
    CHECK(g.attributes[1].n_elem == 3);
    CHECK(std::memcmp(g.attributes[1].value, position, 12) == 0);
    REQUIRE(g.links.size() == 1);
    CHECK(g.links[0].target == 1000);

    // Group without attributes: no Attribute Info message
    hdr.clear();
    hdr.link_info(true, 0);
    hdr.group_info();
    c.image.clear();
    hdr.emit(c.image, 2000, 20);
    h5::parse_chunk(c);
    CHECK(types(c) == std::vector<int>({0x02, 0x0A, 0x00}));
    CHECK(c.free_space == 20);

    // Dataset: Dataspace, Datatype (0x01), Fill Value (0x01), Data Layout; path_gain with 5 paths has 85 bytes
    test_file f("test_qd_hdf5_messages.bin");
    h5::file file;
    REQUIRE(file.open(f.fn, "wb"));
    h5::output out(file, 48);
    const uint64_t n = 5;
    const double gain[5] = {1, 2, 3, 4, 5};
    h5::emit_dataset(out, h5::float32, 1, &n, gain, 20);
    CHECK(out.buf.size() == 85);
    c.image = out.buf;
    c.addr = 48;
    h5::parse_chunk(c);
    CHECK(types(c) == std::vector<int>({0x01, 0x03, 0x05, 0x08}));
    REQUIRE(c.messages.size() == 4);
    CHECK(c.messages[0].flags == 0x00);
    CHECK(c.messages[1].flags == 0x01);
    CHECK(c.messages[2].flags == 0x01);
    CHECK(c.messages[3].flags == 0x00);
    CHECK(c.data(c.messages[2])[1] == 0x09); // Compact: allocation early
    o.chunks.assign(1, c);
    const h5::dataset d = h5::decode_dataset(o, 1000);
    CHECK(d.compact);
    CHECK(d.size == 20);
    float stored[5];
    std::memcpy(stored, d.data, 20);
    CHECK(stored[4] == 5.0f);
}

TEST_CASE("qd_hdf5 - chunks follow the tail rule")
{
    // A Link message of 20 bytes and free space F: nothing (0), a gap (1-3) or one NIL message (4 and more)
    std::vector<uint8_t> link;
    h5::link_message(link, "abcde", 1234);
    REQUIRE(link.size() == 20);

    QD_DIAG("\n# diag: chunk tail by free space\n");
    QD_DIAG("free,ohdr,length,messages\n");
    for (size_t free_space : {0, 1, 2, 3, 4, 20})
        for (bool ohdr : {true, false})
        {
            INFO("free space " << free_space << ", OHDR " << ohdr);
            h5::chunk c;
            c.ohdr = ohdr, c.addr = 4096;
            h5::encode_chunk(c.image, ohdr, link.data(), link.size(), link.size() + free_space);
            h5::parse_chunk(c);
            QD_DIAG("%d,%d,%d,%d\n", (int)free_space, (int)ohdr, (int)c.image.size(), (int)c.messages.size());
            CHECK(c.image.size() == (ohdr ? 7 : 4) + link.size() + free_space + 4);
            CHECK(c.free_space == free_space);
            CHECK(c.messages.size() == (free_space >= 4 ? 2u : 1u));
            if (free_space >= 4)
                CHECK(c.messages[1].size == free_space - 4);
            if (ohdr) // Body size field
                CHECK(c.image[6] == link.size() + free_space);
            CHECK(h5::get_le(&c.image[c.image.size() - 4], 4) == h5::lookup3(c.image.data(), c.image.size() - 4));

            // Any flipped byte breaks the checksum or the prefix
            for (size_t i = 0; i < c.image.size(); i += 7)
            {
                h5::chunk bad = c;
                bad.image[i] ^= 0x01;
                CHECK_THROWS_AS(h5::parse_chunk(bad), std::invalid_argument);
            }
        }

    // A gap together with a NIL message is rejected
    h5::chunk c;
    c.ohdr = false;
    const std::vector<uint8_t> nil = bytes({0x00, 0x00, 0x00, 0x00});
    std::vector<uint8_t> msgs = link;
    msgs.insert(msgs.end(), nil.begin(), nil.end());
    h5::encode_chunk(c.image, false, msgs.data(), msgs.size(), msgs.size() + 2);
    CHECK_THROWS_AS(h5::parse_chunk(c), std::invalid_argument);
}

TEST_CASE("qd_hdf5 - header::emit splits at 65535 bytes and keeps the reserve in the last chunk")
{
    // Link Info (22) + Group Info (6) + one link of 16 + n bytes (name of n >= 256 bytes) + reserve 19 = 65535
    for (size_t extra : {0, 1})
    {
        INFO("extra " << extra);
        const size_t n = 65535 - 22 - 6 - 19 - 16 + extra;
        h5::header hdr;
        hdr.link_info(false);
        hdr.group_info();
        hdr.link(std::string(n, 'x'), 777);
        std::vector<uint8_t> block;
        hdr.emit(block, 48, 19);
        CHECK(block.size() == hdr.size(19));
        h5::chunk c0;
        c0.addr = 48;
        const size_t c0_length = extra == 0 ? 8 + 65535 + 4 : 7 + 48 + 4;
        c0.image.assign(block.begin(), block.begin() + c0_length);
        h5::parse_chunk(c0);
        if (extra == 0) // Fits exactly: one chunk without free space beyond the reserve
        {
            CHECK(block.size() == c0_length);
            CHECK(c0.free_space == 19);
        }
        else // One byte more: Link Info and Group Info stay in chunk 0, the link moves to the next chunk
        {
            CHECK(types(c0) == std::vector<int>({0x02, 0x0A, 0x10}));
            CHECK(c0.free_space == 0);
            const auto &cont = c0.messages[2];
            CHECK(h5::get_le(c0.data(cont), 8) == 48 + c0_length); // Directly after chunk 0
            h5::chunk c1;
            c1.ohdr = false, c1.addr = 48 + c0_length;
            c1.image.assign(block.begin() + c0_length, block.end());
            CHECK(h5::get_le(c0.data(cont) + 8, 8) == c1.image.size());
            h5::parse_chunk(c1);
            CHECK(types(c1) == std::vector<int>({0x06, 0x00}));
            CHECK(c1.free_space == 19);
        }
    }

    // Many links: all messages come back in order from read_object
    test_file f("test_qd_hdf5_split.bin");
    h5::file file;
    REQUIRE(file.open(f.fn, "wb"));
    h5::header hdr;
    hdr.link_info(true, 300);
    hdr.group_info();
    for (int i = 0; i < 300; ++i)
        hdr.link(std::string(500, (char)('a' + i % 26)) + std::to_string(i), 100 + i, i);
    h5::output out(file, 48);
    const uint64_t addr = hdr.emit(out.buf, out.base, 20);
    out.flush();
    file.close();
    REQUIRE(file.open(f.fn, "rb"));
    h5::object o = h5::read_object(file, addr, out.addr());
    QD_DIAG("\n# diag: split header with 300 links of 525 bytes\n");
    QD_DIAG("chunk,addr,length,free\n");
    for (size_t k = 0; k < o.chunks.size(); ++k)
        QD_DIAG("%d,%llu,%llu,%d\n", (int)k, (unsigned long long)o.chunks[k].addr, (unsigned long long)o.chunks[k].length(), (int)o.chunks[k].free_space);
    CHECK(o.chunks.size() == 3);
    for (size_t k = 0; k + 1 < o.chunks.size(); ++k)
        CHECK(o.chunks[k].addr + o.chunks[k].length() == o.chunks[k + 1].addr);
    CHECK(o.chunks.back().free_space == 20);
    const h5::group g = h5::decode_group(o);
    REQUIRE(g.links.size() == 300);
    for (int i = 0; i < 300; ++i)
    {
        CHECK(g.links[i].target == (uint64_t)(100 + i));
        CHECK(g.links[i].corder == (uint64_t)i);
    }
}

TEST_CASE("qd_hdf5 - chain_links grows groups in place and by continuation chunks")
{
    test_file f("test_qd_hdf5_chain.hdf5");
    quadriga_lib::hdf5_create(f.fn, 2);
    h5::file file;
    REQUIRE(file.open(f.fn, "r+b"));
    h5::superblock sb = h5::read_superblock(file);

    // A group with creation order and no links, and a channel-like group with room for exactly one 'Data' link
    h5::header hdr;
    h5::output out(file, sb.eof);
    hdr.link_info(true, 0);
    hdr.group_info();
    const uint64_t data_addr = hdr.emit(out.buf, out.base, 20);
    hdr.clear();
    hdr.link_info(false);
    hdr.group_info();
    const uint32_t zero = 0;
    hdr.attribute("NumSnap", h5::uint32, 0, nullptr, &zero);
    const uint64_t channel_addr = hdr.emit(out.buf, out.base, h5::link_message_size(4, false));
    commit(file, out, sb.root);
    sb = h5::read_superblock(file);

    // Channel group: the 'Data' link fills the reserve exactly
    {
        h5::chain_links g;
        g.load(file, channel_addr, sb.eof);
        CHECK_NOTHROW(g.check_insert(4, false));
        CHECK_THROWS_AS(g.check_insert(5, false), std::invalid_argument);
        CHECK_THROWS_AS(g.check_insert(4, true), std::invalid_argument); // No creation order
        h5::output o(file, sb.eof);
        commit(file, o, sb.root, g.prepare_insert("Data", data_addr, false, o.buf, o.base));
        sb = h5::read_superblock(file);
        h5::object obj = h5::read_object(file, channel_addr, sb.eof);
        CHECK(obj.chunks.back().free_space == 0);
        REQUIRE(h5::decode_group(obj).find("Data") != nullptr);
        CHECK(h5::decode_group(obj).find("Data")->target == data_addr);
    }

    // Group with creation order: links are added until three chunks exist
    QD_DIAG("\n# diag: growth of a group with creation order\n");
    QD_DIAG("links,chunks,last_body,last_free\n");
    size_t n_links = 0, n_chunks = 1;
    while (n_chunks < 3)
    {
        h5::chain_links g;
        g.load(file, data_addr, sb.eof);
        const std::string name = "par_" + std::to_string(n_links) + std::string(20, 'p');
        REQUIRE_NOTHROW(g.check_insert(name.size(), true));
        h5::output o(file, sb.eof);
        commit(file, o, sb.root, g.prepare_insert(name, 5000 + n_links, true, o.buf, o.base));
        sb = h5::read_superblock(file);
        ++n_links;

        h5::object obj = h5::read_object(file, data_addr, sb.eof);
        const h5::group view = h5::decode_group(obj);
        if (obj.chunks.size() != n_chunks)
            QD_DIAG("%d,%d,%d,%d\n", (int)n_links, (int)obj.chunks.size(), (int)obj.chunks.back().body(), (int)obj.chunks.back().free_space);
        n_chunks = obj.chunks.size();
        CHECK(obj.chunks.back().free_space >= 20);
        CHECK(view.max_corder == n_links);
        REQUIRE(view.links.size() == n_links);
        for (size_t i = 0; i < n_links; ++i) // Message order = creation order
        {
            CHECK(view.links[i].corder == i);
            CHECK(view.links[i].target == 5000 + i);
        }
    }
    CHECK(n_chunks == 3);
}

TEST_CASE("qd_hdf5 - root_links builds the directory of link chunks")
{
    test_file f("test_qd_hdf5_root.hdf5");
    quadriga_lib::hdf5_create(f.fn, 1);
    h5::file file;
    REQUIRE(file.open(f.fn, "r+b"));

    // Adds a link to a new 16-byte object at the end of the block; returns its address
    auto add_link = [&](uint32_t id)
    {
        const h5::superblock sb = h5::read_superblock(file);
        const h5::chunk c0 = h5::read_chunk(file, sb.root, 0, sb.eof);
        h5::root_links::check_chunk0(c0, 4);
        h5::root_links root("channel_");
        root.load(file, c0, sb.eof, 0);
        REQUIRE(root.last_id() == id - 1);
        const std::string name = "channel_" + std::to_string(id);
        h5::output out(file, sb.eof);
        const uint64_t target = out.addr() + root.insert_size(name.size());
        const h5::patch p = root.prepare_insert(name, target, out.buf, out.base);
        REQUIRE(out.addr() == target);
        out.append(nullptr, 16);
        commit(file, out, sb.root, {p});
        return target;
    };

    // 9000 links cross from root chunk 0 into the first directory chunk
    std::vector<uint64_t> targets;
    for (uint32_t id = 1; id <= 9000; ++id)
        targets.push_back(add_link(id));

    h5::superblock sb = h5::read_superblock(file);
    const h5::chunk c0 = h5::read_chunk(file, sb.root, 0, sb.eof);
    std::vector<std::pair<uint64_t, uint64_t>> entries;
    for (const auto &m : c0.messages)
        if (m.type == h5::msg_continuation)
            entries.push_back({h5::get_le(c0.data(m), 8), h5::get_le(c0.data(m) + 8, 8)});
    REQUIRE(entries.size() == h5::root_directory_entries);
    CHECK(c0.free_space == 0);

    // Link chunk sizes 1024, 2048, 4096, then the maximum; the 16th entry is a directory chunk
    QD_DIAG("\n# diag: root directory after 9000 links\n");
    QD_DIAG("entry,addr,length,links\n");
    size_t n_found = 0;
    for (size_t k = 0; k < 15; ++k)
    {
        CHECK(entries[k].second == 8 + std::min((size_t)1024 << std::min(k, (size_t)4), h5::link_chunk_body_max));
        const h5::chunk lc = h5::read_chunk(file, entries[k].first, entries[k].second, sb.eof);
        size_t n = 0;
        for (const auto &m : lc.messages)
            n += m.type == h5::msg_link;
        n_found += n;
        QD_DIAG("%d,%llu,%llu,%d\n", (int)k, (unsigned long long)entries[k].first, (unsigned long long)entries[k].second, (int)n);
    }
    CHECK(entries[15].second == 8 + h5::directory_chunk_body);
    const h5::chunk dir = h5::read_chunk(file, entries[15].first, entries[15].second, sb.eof);
    CHECK(dir.messages[0].type == h5::msg_continuation);
    CHECK(n_found < 9000);

    // Every link is found by the binary search and removed; the others keep their order
    for (size_t i : {(size_t)0, (size_t)1, (size_t)4000, (size_t)7310, (size_t)8998})
    {
        INFO("link " << i);
        sb = h5::read_superblock(file);
        h5::chunk root0 = h5::read_chunk(file, sb.root, 0, sb.eof);
        h5::root_links root("channel_");
        root.load(file, root0, sb.eof, targets[i]);
        h5::patch p;
        REQUIRE(root.prepare_remove(targets[i], p));
        CHECK(!root.prepare_remove(targets[i], p)); // Removed from the image
        file.write(p.addr, p.bytes.data(), p.bytes.size());
        h5::chunk lc = h5::read_chunk(file, p.addr, p.bytes.size(), sb.eof);
        uint64_t previous = 0;
        for (const auto &m : lc.messages)
            if (m.type == h5::msg_link)
            {
                h5::link l;
                REQUIRE(h5::decode_link(lc.data(m), m.size, l));
                CHECK(l.target != targets[i]);
                CHECK(l.target > previous);
                previous = l.target;
            }
    }

    // The last link names the largest ID ever linked
    sb = h5::read_superblock(file);
    h5::root_links root("channel_");
    root.load(file, h5::read_chunk(file, sb.root, 0, sb.eof), sb.eof, 0);
    CHECK(root.last_id() == 9000);
    h5::patch p;
    CHECK(!root.prepare_remove(48, p)); // No link chunk below the root
}

TEST_CASE("qd_hdf5 - file reads are bounds-checked and the header window returns the file bytes")
{
    test_file f("test_qd_hdf5_file.bin");
    std::vector<uint8_t> content(200000);
    for (size_t i = 0; i < content.size(); ++i)
        content[i] = (uint8_t)(i * 7 + i / 251);
    {
        h5::file w;
        REQUIRE(w.open(f.fn, "wb"));
        w.write(0, content.data(), content.size());
        w.close();
    }

    h5::file file;
    REQUIRE(file.open(f.fn, "r+b"));
    CHECK(file.size() == content.size());
    uint8_t buf[70000];

    // Outside the file, also where addr + n would wrap
    CHECK_THROWS_AS(file.read(content.size() - 3, buf, 4), std::invalid_argument);
    CHECK_THROWS_AS(file.read(h5::undef, buf, 16), std::invalid_argument);
    CHECK_THROWS_AS(file.read(1ULL << 63, buf, 16), std::invalid_argument);
    CHECK_THROWS_AS(file.read_header(1ULL << 63, buf, (size_t)1 << 20), std::invalid_argument);
    CHECK_THROWS_AS(file.read_header(content.size() - 3, buf, 4), std::invalid_argument);

    // Header reads inside, across the end of and outside the window, and longer than the window
    for (auto r : std::vector<std::pair<uint64_t, size_t>>{{1000, 100}, {1100, 300}, {1000 + 65530, 20}, {150000, 8},
                                                          {150004, 40}, {199990, 10}, {10, 65600}, {12, 3}})
    {
        INFO("addr " << r.first << ", n " << r.second);
        file.read_header(r.first, buf, r.second);
        CHECK(std::memcmp(buf, content.data() + r.first, r.second) == 0);
        file.read(r.first, buf, r.second);
        CHECK(std::memcmp(buf, content.data() + r.first, r.second) == 0);
    }

    // A write drops the window
    file.read_header(5000, buf, 16);
    const uint8_t ones[4] = {1, 1, 1, 1};
    file.write(5004, ones, 4);
    file.read_header(5000, buf, 16);
    CHECK(std::memcmp(buf + 4, ones, 4) == 0);
    file.close();

    // UTF-8 file names (passed through on Linux)
    test_file g("test_qd_hdf5_\xc3\xa4\xc3\xb6\xe2\x82\xac.bin");
    CHECK(!h5::file::exists(g.fn));
    {
        h5::file w;
        REQUIRE(w.open(g.fn, "wb"));
        w.write(0, content.data(), 10);
        w.close();
    }
    CHECK(h5::file::exists(g.fn));
    h5::file r;
    REQUIRE(r.open(g.fn, "rb"));
    CHECK(r.size() == 10);

#if !defined(_WIN32)
    // 64-bit offsets: a sparse file with data at 2^32 + 16
    test_file big("test_qd_hdf5_4gb.bin");
    {
        h5::file w;
        REQUIRE(w.open(big.fn, "wb"));
        w.write((1ULL << 32) + 16, content.data(), 64);
        w.close();
    }
    h5::file b;
    REQUIRE(b.open(big.fn, "rb"));
    CHECK(b.size() == (1ULL << 32) + 80);
    b.read((1ULL << 32) + 16, buf, 64);
    CHECK(std::memcmp(buf, content.data(), 64) == 0);
    b.read_header((1ULL << 32) + 20, buf, 8);
    CHECK(std::memcmp(buf, content.data() + 4, 8) == 0);
#endif
}

TEST_CASE("qd_hdf5 - decimal names and the superblock checks")
{
    CHECK(h5::parse_decimal("0") == 0);
    CHECK(h5::parse_decimal("17") == 17);
    CHECK(h5::parse_decimal("4294967295") == 4294967295LL);
    CHECK(h5::parse_decimal("4294967296") == -1);
    CHECK(h5::parse_decimal("07") == -1);
    CHECK(h5::parse_decimal("") == -1);
    CHECK(h5::parse_decimal("1a") == -1);
    CHECK(h5::parse_decimal("12345678901") == -1);

    test_file f("test_qd_hdf5_superblock.bin");
    auto check_superblock = [&](const std::vector<uint8_t> &sb, size_t file_size)
    {
        {
            h5::file w;
            REQUIRE(w.open(f.fn, "wb"));
            std::vector<uint8_t> data = sb;
            data.resize(file_size, 0);
            w.write(0, data.data(), data.size());
            w.close();
        }
        h5::file r;
        REQUIRE(r.open(f.fn, "rb"));
        return h5::read_superblock(r);
    };
    const h5::superblock sb = check_superblock(h5::superblock_bytes(1000, 48), 1000);
    CHECK(sb.eof == 1000);
    CHECK(sb.root == 48);
    CHECK_NOTHROW(check_superblock(h5::superblock_bytes(1000, 48), 5000));                         // Bytes after EOF
    CHECK_THROWS_AS(check_superblock(h5::superblock_bytes(1000, 48), 999), std::invalid_argument);  // EOF > size
    CHECK_THROWS_AS(check_superblock(h5::superblock_bytes(1000, 1000), 1000), std::invalid_argument); // Root at EOF
    CHECK_THROWS_AS(check_superblock(bytes({0x89, 'H', 'D'}), 3), std::invalid_argument);
    auto sb0 = h5::superblock_bytes(1000, 48);
    sb0[8] = 0; // Version 0 (written by the HDF5 library by default)
    CHECK_THROWS_AS(check_superblock(sb0, 1000), std::invalid_argument);
    sb0 = h5::superblock_bytes(1000, 48);
    sb0[40] ^= 1; // Checksum
    CHECK_THROWS_AS(check_superblock(sb0, 1000), std::invalid_argument);
}
