// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Native reader and writer for the subset of HDF5 used by the quadriga-lib channel files (private, header-only)
//
// Purpose
// - The channel file functions (qd_channel_hdf5.cpp) write and read HDF5 files without the HDF5 library. This header
//   holds the file-format layer: binary encoding and decoding of the structures listed below, checksums, bounds-checked
//   file access, and the two kinds of group headers that grow after they were written. The channel schema (object
//   names, attributes, shapes) lives in qd_channel_hdf5.cpp.
//
// Specification followed: "HDF5 File Format Specification Version 3.0" (The HDF Group). Section names used here:
// "Superblock" (version 2), "Version 2 Object Header", "Header Message Types": NIL, Dataspace (version 2), Link Info
// (version 0), Datatype (version 1: fixed-point, floating-point, string, reference classes), Fill Value (version 3),
// Link (version 1, hard links), Data Layout (version 3, compact and contiguous), Group Info (version 0), Attribute
// (version 3), Object Header Continuation, Attribute Info (version 0); "Appendix C: Jenkins' lookup3 Hash Function".
//
// Supported subset (the writer produces only these structures; the reader accepts only these and throws
// std::invalid_argument for anything else, e.g. older superblocks, version 1 object headers, dense link or attribute
// storage, chunked layouts, filters, variable-length or NUL-padded strings, compound types, shared messages):
// - Superblock version 2 at address 0 with 8-byte offsets and lengths, no superblock extension.
// - Version 2 object headers: chunk 0 ("OHDR") and continuation chunks ("OCHK"), each with a lookup3 checksum and a
//   body of at most 65535 bytes. Free space at the end of a chunk is either absent, a gap of 1 to 3 zero bytes, or
//   exactly one NIL message (never two NIL messages, never a gap and a NIL message in one chunk).
// - Datasets: scalar or simple dataspace of rank 1 to 3 without maximum extents; element types float32, float64,
//   (u)int32, (u)int64 (little-endian), fixed-length ASCII strings with NUL-terminated padding, object references.
//   Raw data of at most 4096 bytes is stored in the header (compact layout, covered by the checksum); larger data is
//   contiguous and follows its header directly (not checksummed).
// - Groups: compact link storage (Link messages in the header), attributes in the header (with an Attribute Info
//   message), optional tracking and indexing of the link creation order.
// - Growable group headers (class chain_links): a group header whose last chunk keeps free space for a new link; when
//   it is full, a new continuation chunk is appended at the end of the file and linked from the last chunk.
// - The root group (class root_links): its chunk 0 holds the links written at creation, followed by a directory of
//   Continuation messages. They point to "link chunks" (continuation chunks holding only links, 1 KB growing to 8 KB)
//   and, in the last entry of a full directory chunk, to the next "directory chunk" (16 KB of Continuation messages).
//   New links are appended to the last link chunk, so a link to an object at address A lies in the link chunk with
//   the largest address below A.
// - Files are append-only: new objects are written at the end-of-file address of the superblock; existing chunks are
//   only rewritten in place (same address and length, new checksum).
//
// Compatibility: every structure used exists since HDF5 1.8 and is at or below the versions the HDF5 library writes
// for 1.8-compatible files, so HDF5 readers from version 1.8 on (h5py, MATLAB, HDFView) can read the files.
//
// Portability: C++17, no compiler builtins, little-endian hosts only (checked at compile time), 64-bit file offsets,
// UTF-8 file names (converted to wide names under Windows). No global or static mutable state: different files can
// be read and written from parallel threads; one writer per file.

#ifndef qd_hdf5_H
#define qd_hdf5_H

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <filesystem>
#include <share.h>
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "qd_hdf5.hpp supports little-endian hosts only."
#endif

namespace qd
{
    namespace hdf5
    {
        // ---- Constants ----
        static constexpr uint64_t undef = ~0ULL;               // Undefined address
        static constexpr size_t superblock_size = 48;          // Superblock version 2; also the lowest object address
        static constexpr size_t max_chunk_body = 65535;        // Free space must fit the 16-bit size of a NIL message
        static constexpr size_t compact_limit = 4096;          // Datasets with at most this many raw bytes are compact
        static constexpr size_t root_directory_entries = 16;   // Continuation messages reserved in root chunk 0
        static constexpr size_t directory_chunk_body = 16380;  // 819 Continuation messages
        static constexpr size_t link_chunk_body_first = 1024;  // Body of link chunk k: min(1024 * 2^k, 8192)
        static constexpr size_t link_chunk_body_max = 8192;    //
        static constexpr size_t chain_chunk_body_max = 16384;  // New chunk of a growable group: max(L + 20, min(16384, 2 * last))
        static constexpr size_t continuation_size = 20;        // Size of a Continuation message
        static constexpr size_t header_window = 16384;         // Read window for object headers of unknown length
        static constexpr size_t header_lookback = 4096;        // The window starts this far before the header
        static constexpr size_t io_max = (size_t)1 << 30;      // Largest single read or write call
        static constexpr size_t io_block = 1048576;            // Output buffer; larger data is written directly
        static constexpr size_t max_name_length = 65000;       // A Link message must fit the 16-bit message size

        // Message types
        enum : uint8_t
        {
            msg_nil = 0x00,
            msg_dataspace = 0x01,
            msg_link_info = 0x02,
            msg_datatype = 0x03,
            msg_fill_value = 0x05,
            msg_link = 0x06,
            msg_layout = 0x08,
            msg_group_info = 0x0A,
            msg_attribute = 0x0C,
            msg_continuation = 0x10,
            msg_attribute_info = 0x15
        };

        // ---- Errors ----
        [[noreturn]] static inline void malformed(const std::string &what)
        {
            throw std::invalid_argument("Malformed HDF5 file: " + what + ".");
        }

        // A valid HDF5 file that uses structures outside the subset (written or modified by other tools)
        [[noreturn]] static inline void unsupported()
        {
            throw std::invalid_argument("Unsupported HDF5 file: quadriga-lib reads only channel files of version 3 that it "
                                        "wrote itself; files written by quadriga-lib 0.12.4 and earlier (version 2) and "
                                        "files modified with other HDF5 tools are not supported.");
        }

        // ---- Checksum ----
        // Jenkins' lookup3 hashlittle() with initval 0, as used by HDF5 for metadata checksums. Byte-wise form (as
        // H5_checksum_lookup3 of the HDF5 library): independent of alignment and host byte order, never reads past
        // the end of the key.
        static inline uint32_t lookup3(const uint8_t *k, size_t length)
        {
            auto rot = [](uint32_t x, int s)
            { return (x << s) ^ (x >> (32 - s)); };
            uint32_t a, b, c;
            a = b = c = 0xdeadbeefu + (uint32_t)length;
            while (length > 12)
            {
                a += k[0] + ((uint32_t)k[1] << 8) + ((uint32_t)k[2] << 16) + ((uint32_t)k[3] << 24);
                b += k[4] + ((uint32_t)k[5] << 8) + ((uint32_t)k[6] << 16) + ((uint32_t)k[7] << 24);
                c += k[8] + ((uint32_t)k[9] << 8) + ((uint32_t)k[10] << 16) + ((uint32_t)k[11] << 24);
                a -= c, a ^= rot(c, 4), c += b;
                b -= a, b ^= rot(a, 6), a += c;
                c -= b, c ^= rot(b, 8), b += a;
                a -= c, a ^= rot(c, 16), c += b;
                b -= a, b ^= rot(a, 19), a += c;
                c -= b, c ^= rot(b, 4), b += a;
                length -= 12, k += 12;
            }
            switch (length) // The last 1 to 12 bytes
            {
            case 12:
                c += (uint32_t)k[11] << 24;
                [[fallthrough]];
            case 11:
                c += (uint32_t)k[10] << 16;
                [[fallthrough]];
            case 10:
                c += (uint32_t)k[9] << 8;
                [[fallthrough]];
            case 9:
                c += k[8];
                [[fallthrough]];
            case 8:
                b += (uint32_t)k[7] << 24;
                [[fallthrough]];
            case 7:
                b += (uint32_t)k[6] << 16;
                [[fallthrough]];
            case 6:
                b += (uint32_t)k[5] << 8;
                [[fallthrough]];
            case 5:
                b += k[4];
                [[fallthrough]];
            case 4:
                a += (uint32_t)k[3] << 24;
                [[fallthrough]];
            case 3:
                a += (uint32_t)k[2] << 16;
                [[fallthrough]];
            case 2:
                a += (uint32_t)k[1] << 8;
                [[fallthrough]];
            case 1:
                a += k[0];
                break;
            default: // Zero bytes left
                return c;
            }
            c ^= b, c -= rot(b, 14);
            a ^= c, a -= rot(c, 11);
            b ^= a, b -= rot(a, 25);
            c ^= b, c -= rot(b, 16);
            a ^= c, a -= rot(c, 4);
            b ^= a, b -= rot(a, 14);
            c ^= b, c -= rot(b, 24);
            return c;
        }

        // ---- Little-endian integers ----
        static inline void put_le(std::vector<uint8_t> &out, uint64_t value, int n_bytes)
        {
            for (int i = 0; i < n_bytes; ++i)
                out.push_back((uint8_t)(value >> (8 * i)));
        }

        static inline void put_le(uint8_t *dst, uint64_t value, int n_bytes)
        {
            for (int i = 0; i < n_bytes; ++i)
                dst[i] = (uint8_t)(value >> (8 * i));
        }

        static inline uint64_t get_le(const uint8_t *src, int n_bytes)
        {
            uint64_t value = 0;
            for (int i = n_bytes - 1; i >= 0; --i)
                value = (value << 8) | src[i];
            return value;
        }

        // True if [addr, addr + n) lies within [lower, end); addr + n is never computed, so no value can wrap
        static inline bool in_bounds(uint64_t addr, uint64_t n, uint64_t end, uint64_t lower = 0)
        {
            return n <= end && addr >= lower && addr <= end - n;
        }

        // a * b without overflow; returns false if the product does not fit 64 bits
        static inline bool multiply(uint64_t a, uint64_t b, uint64_t &product)
        {
            if (a != 0 && b > UINT64_MAX / a)
                return false;
            product = a * b;
            return true;
        }

        // Value of a decimal number of 1 to 10 digits without a leading zero ("0" itself is allowed) that is at most
        // 2^32 - 1; -1 for anything else
        static inline long long parse_decimal(std::string_view s)
        {
            if (s.empty() || s.size() > 10 || (s[0] == '0' && s.size() > 1))
                return -1;
            long long value = 0;
            for (char ch : s)
            {
                if (ch < '0' || ch > '9')
                    return -1;
                value = value * 10 + (ch - '0');
            }
            return value <= 4294967295LL ? value : -1;
        }

        // ---- Element types ----
        enum class type_t : uint8_t
        {
            f32,
            f64,
            u64,
            i64,
            u32,
            i32,
            str, // Fixed-length string of 'size' bytes, NUL-terminated padding, ASCII
            ref  // Object reference (address of an object header)
        };

        struct dtype
        {
            type_t t;
            uint32_t size; // Element size in bytes
        };

        static constexpr dtype float32 = {type_t::f32, 4}, float64 = {type_t::f64, 8}, uint64 = {type_t::u64, 8},
                               int64 = {type_t::i64, 8}, uint32 = {type_t::u32, 4}, int32 = {type_t::i32, 4},
                               object_reference = {type_t::ref, 8};

        static inline dtype string_type(uint32_t size) { return {type_t::str, size}; }

        static inline bool operator==(const dtype &a, const dtype &b) { return a.t == b.t && a.size == b.size; }
        static inline bool operator!=(const dtype &a, const dtype &b) { return !(a == b); }

        // Size of the Datatype message data of a type
        static inline size_t datatype_size(dtype t)
        {
            return (t.t == type_t::f32 || t.t == type_t::f64) ? 20 : (t.t == type_t::str || t.t == type_t::ref) ? 8
                                                                                                                 : 12;
        }

        // Datatype messages (version 1) of the types other than strings, in the order of type_t, byte-identical to the
        // HDF5 library's encoding of the predefined types (datatype_size bytes each). Fixed-point: class 0x10, bit 3 of
        // the bit field = signed, size, offset 0, precision. Floating-point: class 0x11, sign bit position, size,
        // offset, precision, exponent location and size, mantissa location and size, exponent bias.
        static inline const uint8_t *datatype_bytes(type_t t)
        {
            static const uint8_t bytes[8][20] = {
                {0x11, 0x20, 0x1f, 0, 4, 0, 0, 0, 0, 0, 32, 0, 23, 8, 0, 23, 0x7f, 0, 0, 0},  // f32
                {0x11, 0x20, 0x3f, 0, 8, 0, 0, 0, 0, 0, 64, 0, 52, 11, 0, 52, 0xff, 3, 0, 0}, // f64
                {0x10, 0x00, 0, 0, 8, 0, 0, 0, 0, 0, 64, 0},                                   // u64
                {0x10, 0x08, 0, 0, 8, 0, 0, 0, 0, 0, 64, 0},                                   // i64
                {0x10, 0x00, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0},                                   // u32
                {0x10, 0x08, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0},                                   // i32
                {},                                                                            // str (encoded below)
                {0x17, 0, 0, 0, 8, 0, 0, 0}};                                                  // ref
            return bytes[(int)t];
        }

        // Datatype message (version 1); strings: class 0x13, NUL-terminated padding, ASCII, size
        static inline void encode_datatype(std::vector<uint8_t> &out, dtype t)
        {
            if (t.t == type_t::str)
            {
                out.insert(out.end(), {0x13, 0, 0, 0});
                put_le(out, t.size, 4);
            }
            else
                out.insert(out.end(), datatype_bytes(t.t), datatype_bytes(t.t) + datatype_size(t));
        }

        // Decodes a Datatype message; accepts exactly the encodings of encode_datatype (strings of any size >= 1)
        static inline bool decode_datatype(const uint8_t *p, size_t n, dtype &t)
        {
            if (n == 8 && p[0] == 0x13 && p[1] == 0 && p[2] == 0 && p[3] == 0)
            {
                t = string_type((uint32_t)get_le(p + 4, 4));
                return t.size != 0;
            }
            for (dtype candidate : {float32, float64, uint64, int64, uint32, int32, object_reference})
                if (n == datatype_size(candidate) && std::memcmp(datatype_bytes(candidate.t), p, n) == 0)
                {
                    t = candidate;
                    return true;
                }
            return false;
        }

        // ---- File access ----
        // A file opened unbuffered, so that every write reaches the operating system in program order. Reads are
        // bounds-checked against the size at open; seek or read failures throw std::invalid_argument, write and close
        // failures std::runtime_error. Object headers of unknown length are read through a forward window of 16 KiB.
        class file
        {
        public:
            file() = default;
            file(const file &) = delete;
            file &operator=(const file &) = delete;
            ~file()
            {
                if (f_ != nullptr)
                    std::fclose(f_);
            }

            // Opens 'fn' (UTF-8) with the mode "rb", "r+b" or "wb"; false if it cannot be opened or sized
            bool open(const std::string &fn, const char *mode)
            {
                f_ = open_utf8(fn, mode);
                if (f_ == nullptr)
                    return false;
                fn_ = fn;
                std::setvbuf(f_, nullptr, _IONBF, 0);
#if defined(_WIN32)
                if (_fseeki64(f_, 0, SEEK_END) != 0)
                    return false;
                long long size = _ftelli64(f_);
#else
                if (std::fseek(f_, 0, SEEK_END) != 0)
                    return false;
                long size = std::ftell(f_);
#endif
                if (size < 0)
                    return false;
                size_ = (uint64_t)size;
                return true;
            }

            // True if the file exists, also if it cannot be read (the open fails with another reason than a missing
            // file or directory)
            static bool exists(const std::string &fn)
            {
                errno = 0;
                FILE *f = open_utf8(fn, "rb");
                if (f == nullptr)
                    return errno != 0 && errno != ENOENT && errno != ENOTDIR;
                std::fclose(f);
                return true;
            }

            uint64_t size() const { return size_; }

            // Reads n bytes at 'addr' directly (in pieces of at most 1 GiB: the Windows C runtime rejects single
            // reads and writes above 2 GiB)
            void read(uint64_t addr, void *dst, size_t n)
            {
                if (!in_bounds(addr, n, size_) || (n != 0 && !seek(addr)))
                    read_error(addr, n);
                for (size_t done = 0; done < n;)
                {
                    const size_t piece = std::min(n - done, io_max);
                    if (std::fread((uint8_t *)dst + done, 1, piece, f_) != piece)
                        read_error(addr, n);
                    done += piece;
                }
            }

            // Reads n bytes at 'addr' through the header window: a read outside the window loads 16 KiB from 4 KiB
            // before 'addr' (a group header lies after its datasets, which are read next), limited to the file
            void read_header(uint64_t addr, void *dst, size_t n)
            {
                if (!in_bounds(addr, n, size_))
                    read_error(addr, n);
                if (!window_.empty() && addr >= window_addr_ && in_bounds(addr - window_addr_, n, window_.size()))
                {
                    std::memcpy(dst, window_.data() + (addr - window_addr_), n);
                    return;
                }
                if (n > header_window)
                {
                    read(addr, dst, n);
                    return;
                }
                window_.clear(); // Invalid while loading
                const uint64_t start = addr - std::min(addr, (uint64_t)(n <= header_window - header_lookback ? header_lookback : 0));
                std::vector<uint8_t> window((size_t)std::min((uint64_t)header_window, size_ - start));
                read(start, window.data(), window.size());
                window_.swap(window);
                window_addr_ = start;
                std::memcpy(dst, window_.data() + (addr - start), n);
            }

            // Writes n bytes at 'addr'; drops the header window
            void write(uint64_t addr, const void *src, size_t n)
            {
                window_.clear();
                if (n != 0 && !seek(addr))
                    throw std::runtime_error("Cannot write file '" + fn_ + "'.");
                for (size_t done = 0; done < n;)
                {
                    const size_t piece = std::min(n - done, io_max);
                    if (std::fwrite((const uint8_t *)src + done, 1, piece, f_) != piece)
                        throw std::runtime_error("Cannot write file '" + fn_ + "'.");
                    done += piece;
                }
                size_ = std::max(size_, addr + n);
            }

            // Closes the file; throws if closing fails (the destructor closes silently)
            void close()
            {
                FILE *f = f_;
                f_ = nullptr;
                if (f != nullptr && std::fclose(f) != 0)
                    throw std::runtime_error("Cannot write file '" + fn_ + "'.");
            }

        private:
            static FILE *open_utf8(const std::string &fn, const char *mode)
            {
#if defined(_WIN32)
                try
                {
                    const std::filesystem::path p = std::filesystem::u8path(fn);
                    const std::wstring wide_mode(mode, mode + std::strlen(mode));
                    return _wfsopen(p.c_str(), wide_mode.c_str(), _SH_DENYNO); // Shared like fopen; not deprecated
                }
                catch (...)
                {
                    return nullptr;
                }
#else
                return std::fopen(fn.c_str(), mode);
#endif
            }

            bool seek(uint64_t addr)
            {
#if defined(_WIN32)
                return addr <= (uint64_t)LLONG_MAX && _fseeki64(f_, (long long)addr, SEEK_SET) == 0;
#else
                static_assert(sizeof(long) >= 8, "64-bit file offsets need a 64-bit 'long'.");
                return addr <= (uint64_t)LONG_MAX && std::fseek(f_, (long)addr, SEEK_SET) == 0;
#endif
            }

            [[noreturn]] static void read_error(uint64_t addr, size_t n)
            {
                malformed("cannot read " + std::to_string(n) + " bytes at address " + std::to_string(addr));
            }

            FILE *f_ = nullptr;
            std::string fn_;
            uint64_t size_ = 0;
            std::vector<uint8_t> window_;
            uint64_t window_addr_ = 0;
        };

        // ---- Superblock version 2 ----
        struct superblock
        {
            uint64_t eof = 0;  // End-of-file address: end of the last allocated byte
            uint64_t root = 0; // Address of the root group's object header
        };

        // Superblock with signature, version 2, 8-byte offsets and lengths, consistency flags 0, base address 0, no
        // extension, and its checksum
        static inline std::vector<uint8_t> superblock_bytes(uint64_t eof, uint64_t root)
        {
            std::vector<uint8_t> b = {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n', 2, 8, 8, 0};
            put_le(b, 0, 8);
            put_le(b, undef, 8);
            put_le(b, eof, 8);
            put_le(b, root, 8);
            put_le(b, lookup3(b.data(), b.size()), 4);
            return b;
        }

        static inline superblock read_superblock(file &f)
        {
            static const uint8_t signature[8] = {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
            uint8_t b[superblock_size];
            const size_t n = (size_t)std::min(f.size(), (uint64_t)superblock_size);
            f.read(0, b, n);
            if (n < 8 || std::memcmp(b, signature, 8) != 0)
                throw std::invalid_argument("Not an HDF5 file.");
            if (n < superblock_size)
                malformed("the superblock is truncated");
            if (b[8] != 2 || b[9] != 8 || b[10] != 8 || get_le(b + 12, 8) != 0 || get_le(b + 20, 8) != undef)
                unsupported();
            if (b[11] > 1) // Bit 0 may be left set by a crashed read-write session of the HDF5 library
                malformed("unknown superblock flags");
            if (get_le(b + 44, 4) != lookup3(b, 44))
                malformed("checksum mismatch at address 0");
            superblock sb;
            sb.eof = get_le(b + 28, 8);
            sb.root = get_le(b + 36, 8);
            if (sb.eof <= superblock_size || sb.eof > f.size())
                malformed("the end-of-file address " + std::to_string(sb.eof) + " does not match the file size " +
                          std::to_string(f.size()));
            if (sb.root < superblock_size || sb.root >= sb.eof)
                malformed("address " + std::to_string(sb.root) + " is outside the file");
            return sb;
        }

        // ---- Object header chunks ----
        struct message_ref
        {
            uint8_t type, flags;
            uint32_t offset; // Position of the message data in the chunk image
            uint32_t size;   // Size of the message data
        };

        // One chunk of a version 2 object header, as read from the file or after a rewrite
        struct chunk
        {
            uint64_t addr = 0;
            bool ohdr = true;                  // Chunk 0 ("OHDR") or a continuation chunk ("OCHK")
            std::vector<uint8_t> image;        // All bytes from the signature to the checksum
            std::vector<message_ref> messages; // In order, NIL messages included
            size_t body_begin = 0, body_end = 0;
            size_t free_space = 0; // Body size minus the sizes of the non-NIL messages

            uint64_t length() const { return image.size(); }
            size_t body() const { return body_end - body_begin; }
            const uint8_t *data(const message_ref &m) const { return image.data() + m.offset; }
        };

        // An in-place rewrite of existing bytes, written after the appended block in a given order
        struct patch
        {
            uint64_t addr = 0;
            std::vector<uint8_t> bytes;
        };

        // Appends a message header; returns its position for end_message
        static inline size_t begin_message(std::vector<uint8_t> &out, uint8_t type)
        {
            // Message flags as the HDF5 library writes them: constant (0x01) for Datatype, Fill Value and Group Info,
            // "do not share" (0x04) for Attribute Info
            const uint8_t flags = (type == msg_datatype || type == msg_fill_value || type == msg_group_info) ? 0x01
                                  : type == msg_attribute_info                                             ? 0x04
                                                                                                           : 0x00;
            const size_t pos = out.size();
            out.insert(out.end(), {type, 0, 0, flags});
            return pos;
        }

        // Sets the data size of the message started at 'pos'
        static inline void end_message(std::vector<uint8_t> &out, size_t pos)
        {
            put_le(&out[pos + 1], out.size() - pos - 4, 2);
        }

        // Link message (version 1, hard link); 'corder' = undef: no creation order
        static inline void link_message(std::vector<uint8_t> &out, std::string_view name, uint64_t target, uint64_t corder = undef)
        {
            const size_t pos = begin_message(out, msg_link);
            const uint8_t width = name.size() < 256 ? 0 : 1;
            out.insert(out.end(), {1, (uint8_t)(width | (corder != undef ? 0x04 : 0x00))});
            if (corder != undef)
                put_le(out, corder, 8);
            put_le(out, name.size(), 1 + width);
            out.insert(out.end(), name.begin(), name.end());
            put_le(out, target, 8);
            end_message(out, pos);
        }

        // Size of a Link message (header included)
        static inline size_t link_message_size(size_t name_length, bool with_corder)
        {
            return 4 + 2 + (with_corder ? 8 : 0) + (name_length < 256 ? 1 : 2) + name_length + 8;
        }

        static inline void continuation_message(std::vector<uint8_t> &out, uint64_t addr, uint64_t length)
        {
            const size_t pos = begin_message(out, msg_continuation);
            put_le(out, addr, 8);
            put_le(out, length, 8);
            end_message(out, pos);
        }

        // Dataspace (version 2): scalar (rank 0) or simple, no maximum extents; data without the message header
        static inline void encode_dataspace(std::vector<uint8_t> &out, int rank, const uint64_t *dims)
        {
            out.insert(out.end(), {2, (uint8_t)rank, 0, (uint8_t)(rank == 0 ? 0 : 1)});
            for (int i = 0; i < rank; ++i)
                put_le(out, dims[i], 8);
        }

        static inline void dataspace_message(std::vector<uint8_t> &out, int rank, const uint64_t *dims)
        {
            const size_t pos = begin_message(out, msg_dataspace);
            encode_dataspace(out, rank, dims);
            end_message(out, pos);
        }

        static inline void datatype_message(std::vector<uint8_t> &out, dtype t)
        {
            const size_t pos = begin_message(out, msg_datatype);
            encode_datatype(out, t);
            end_message(out, pos);
        }

        // Fill Value message (version 3) as the HDF5 library writes it: allocation time early for compact datasets,
        // late for contiguous ones, fill value written "if set", no fill value defined
        static inline void fill_value_message(std::vector<uint8_t> &out, bool compact)
        {
            const size_t pos = begin_message(out, msg_fill_value);
            out.insert(out.end(), {3, (uint8_t)(compact ? 0x09 : 0x0a)});
            end_message(out, pos);
        }

        // Appends a chunk: prefix, the encoded messages, the tail (nothing, a gap of 1-3 zero bytes, or one NIL
        // message filling the free space) and the checksum. 'width_code' selects the size field of chunk 0
        // (0: 1 byte, 1: 2 bytes; -1: by body size).
        static inline void encode_chunk(std::vector<uint8_t> &out, bool ohdr, const uint8_t *msgs, size_t n, size_t body,
                                        int width_code = -1)
        {
            const size_t start = out.size();
            if (ohdr)
            {
                const uint8_t code = (uint8_t)(width_code >= 0 ? width_code : body < 256 ? 0
                                                                                           : 1);
                out.insert(out.end(), {'O', 'H', 'D', 'R', 2, code});
                put_le(out, body, 1 + code);
            }
            else
                out.insert(out.end(), {'O', 'C', 'H', 'K'});
            out.insert(out.end(), msgs, msgs + n);
            const size_t free_space = body - n;
            if (free_space >= 4)
            {
                out.insert(out.end(), {msg_nil, 0, 0, 0});
                put_le(&out[out.size() - 3], free_space - 4, 2);
            }
            out.resize(out.size() + (free_space >= 4 ? free_space - 4 : free_space), 0);
            put_le(out, lookup3(out.data() + start, out.size() - start), 4);
        }

        // Verifies a chunk image (signature, prefix, size and, with 'verify_checksum', the checksum) and lists its messages
        static inline void parse_chunk(chunk &c, bool verify_checksum = true)
        {
            const std::vector<uint8_t> &im = c.image;
            auto bad = [&](const std::string &what)
            { malformed(what + " in the object header chunk at address " + std::to_string(c.addr)); };
            size_t prefix = 4, body = 0;
            if (c.ohdr)
            {
                if (im.size() < 12 || std::memcmp(im.data(), "OHDR", 4) != 0 || im[4] != 2)
                    malformed("no object header at address " + std::to_string(c.addr));
                if ((im[5] & 0xfc) != 0 || (im[5] & 0x03) > 1)
                    bad("unsupported object header flags");
                prefix = 7 + (im[5] & 0x03);
                body = (size_t)get_le(&im[6], 1 + (im[5] & 0x03));
                if (body == 0 || prefix + body + 4 != im.size())
                    bad("invalid chunk size");
            }
            else
            {
                if (im.size() < 8 || im.size() > max_chunk_body + 8 || std::memcmp(im.data(), "OCHK", 4) != 0)
                    malformed("no continuation chunk at address " + std::to_string(c.addr));
                body = im.size() - 8;
            }
            if (verify_checksum && get_le(&im[prefix + body], 4) != lookup3(im.data(), prefix + body))
                malformed("checksum mismatch at address " + std::to_string(c.addr));

            c.messages.clear();
            c.body_begin = prefix, c.body_end = prefix + body;
            size_t pos = prefix, used = 0;
            bool has_nil = false;
            while (c.body_end - pos >= 4)
            {
                const uint8_t type = im[pos], flags = im[pos + 3];
                const size_t size = (size_t)get_le(&im[pos + 1], 2);
                if (size > c.body_end - pos - 4)
                    bad("message size beyond the chunk");
                if ((flags & 0x02) != 0)
                    bad("shared message");
                switch (type)
                {
                case msg_nil:
                case msg_dataspace:
                case msg_link_info:
                case msg_datatype:
                case msg_fill_value:
                case msg_link:
                case msg_layout:
                case msg_group_info:
                case msg_attribute:
                case msg_continuation:
                case msg_attribute_info:
                    break;
                default:
                    bad("unsupported message type " + std::to_string(type));
                }
                c.messages.push_back({type, flags, (uint32_t)(pos + 4), (uint32_t)size});
                if (type == msg_nil)
                    has_nil = true;
                else
                    used += 4 + size;
                pos += 4 + size;
            }
            if (pos != c.body_end && has_nil)
                bad("gap and NIL message");
            c.free_space = body - used;
        }

        // Reads and verifies a chunk; 'length' = 0 for chunk 0 (its size follows from the prefix), otherwise the
        // length of a continuation chunk as given by its Continuation message
        static inline chunk read_chunk(file &f, uint64_t addr, uint64_t length, uint64_t eof)
        {
            chunk c;
            c.addr = addr;
            c.ohdr = length == 0;
            auto outside = [&]()
            { malformed("address " + std::to_string(addr) + " is outside the file"); };
            if (c.ohdr)
            {
                uint8_t prefix[8];
                if (!in_bounds(addr, 8, eof, superblock_size))
                    outside();
                f.read_header(addr, prefix, 8);
                if (std::memcmp(prefix, "OHDR", 4) != 0 || prefix[4] != 2)
                    malformed("no object header at address " + std::to_string(addr));
                if ((prefix[5] & 0xfc) != 0 || (prefix[5] & 0x03) > 1)
                    malformed("unsupported object header flags at address " + std::to_string(addr));
                length = 7 + (prefix[5] & 0x03) + get_le(&prefix[6], 1 + (prefix[5] & 0x03)) + 4;
            }
            else if (length < 8 || length > max_chunk_body + 8)
                malformed("invalid continuation chunk length " + std::to_string(length));
            if (!in_bounds(addr, length, eof, superblock_size))
                outside();
            c.image.resize((size_t)length);
            if (c.ohdr)
                f.read_header(addr, c.image.data(), c.image.size());
            else
                f.read(addr, c.image.data(), c.image.size());
            parse_chunk(c);
            return c;
        }

        // The encoded non-NIL messages of a chunk in order, without the message 'skip' (if given)
        static inline std::vector<uint8_t> chunk_messages(const chunk &c, const message_ref *skip = nullptr)
        {
            std::vector<uint8_t> out;
            out.reserve(c.body());
            for (const message_ref &m : c.messages)
                if (m.type != msg_nil && &m != skip)
                    out.insert(out.end(), c.image.begin() + (m.offset - 4), c.image.begin() + (m.offset + m.size));
            return out;
        }

        // Re-encodes a chunk in place (same address, length and size field) with the given messages and the tail
        static inline void rewrite_chunk(chunk &c, const std::vector<uint8_t> &msgs)
        {
            std::vector<uint8_t> image;
            image.reserve(c.image.size());
            encode_chunk(image, c.ohdr, msgs.data(), msgs.size(), c.body(), c.ohdr ? (c.image[5] & 0x03) : -1);
            c.image.swap(image);
            parse_chunk(c, false);
        }

        // All chunks of an object header in reading order: chunk 0, then the continuation chunks in the order their
        // Continuation messages are found. Each continuation chunk must start at or after the end of the chunk read
        // before it, which rules out cycles and bounds the bytes read by the file size.
        struct object
        {
            std::vector<chunk> chunks;
        };

        static inline object read_object(file &f, uint64_t addr, uint64_t eof)
        {
            object o;
            o.chunks.push_back(read_chunk(f, addr, 0, eof));
            std::vector<std::pair<uint64_t, uint64_t>> queue; // Address and length of the continuation chunks
            uint64_t next = addr + o.chunks[0].length();
            for (size_t k = 0; k < o.chunks.size(); ++k)
            {
                for (const message_ref &m : o.chunks[k].messages)
                    if (m.type == msg_continuation)
                    {
                        if (m.size != 16)
                            malformed("invalid continuation message at address " + std::to_string(o.chunks[k].addr));
                        queue.push_back({get_le(o.chunks[k].data(m), 8), get_le(o.chunks[k].data(m) + 8, 8)});
                    }
                if (k < queue.size())
                {
                    if (queue[k].first < next)
                        malformed("continuation chunk at address " + std::to_string(queue[k].first) + " is out of order");
                    o.chunks.push_back(read_chunk(f, queue[k].first, queue[k].second, eof));
                    next = queue[k].first + queue[k].second;
                }
            }
            return o;
        }

        // Address and length of every chunk of an object
        static inline std::vector<std::pair<uint64_t, uint64_t>> chunk_ranges(const object &o)
        {
            std::vector<std::pair<uint64_t, uint64_t>> r;
            for (const chunk &c : o.chunks)
                r.push_back({c.addr, c.length()});
            return r;
        }

        // ---- Decoded views of objects ----
        // Views point into the chunk images of the object and are valid while the object lives unchanged.

        struct link
        {
            std::string_view name;
            uint64_t target = 0;
            uint64_t corder = undef; // Creation order; undef if not stored
        };

        struct attribute
        {
            std::string_view name;
            dtype type{};
            int rank = 0;
            uint64_t dims[3] = {1, 1, 1};
            uint64_t n_elem = 1;
            const uint8_t *value = nullptr; // n_elem * type.size bytes
        };

        struct group
        {
            std::vector<link> links;
            std::vector<attribute> attributes;
            uint8_t link_info_flags = 0; // 0 or 3 (creation order tracked and indexed)
            uint64_t max_corder = 0;     // Number of links ever created, if the creation order is tracked

            const link *find(std::string_view name) const
            {
                for (const link &l : links)
                    if (l.name == name)
                        return &l;
                return nullptr;
            }
            const attribute *find_attribute(std::string_view name) const
            {
                for (const attribute &a : attributes)
                    if (a.name == name)
                        return &a;
                return nullptr;
            }
        };

        struct dataset
        {
            dtype type{};
            int rank = 0;                 // 0 = scalar
            uint64_t dims[3] = {1, 1, 1}; // Extents, slowest dimension first
            uint64_t n_elem = 1;
            bool compact = true;
            const uint8_t *data = nullptr; // Compact data
            uint64_t address = undef;      // Contiguous data
            uint64_t size = 0;             // Raw data size in bytes
        };

        // Link message (version 1, hard link)
        static inline bool decode_link(const uint8_t *p, size_t n, link &l)
        {
            if (n < 2 || p[0] != 1 || (p[1] & 0xf8) != 0)
                return false;
            const size_t width = (size_t)1 << (p[1] & 0x03);
            size_t pos = 2;
            l.corder = undef;
            if ((p[1] & 0x04) != 0)
            {
                if (n < pos + 8)
                    return false;
                l.corder = get_le(p + pos, 8), pos += 8;
            }
            if (n < pos + width)
                return false;
            const uint64_t name_length = get_le(p + pos, (int)width);
            pos += width;
            if (name_length == 0 || name_length > n - pos || n - pos - name_length != 8)
                return false;
            l.name = std::string_view((const char *)p + pos, (size_t)name_length);
            l.target = get_le(p + pos + name_length, 8);
            return true;
        }

        // Dataspace message (version 2): scalar or simple with rank 1 to 3, no maximum extents
        static inline bool decode_dataspace(const uint8_t *p, size_t n, int &rank, uint64_t *dims, uint64_t &n_elem)
        {
            if (n < 4 || p[0] != 2 || p[2] != 0 || p[1] > 3 || n != 4 + 8 * (size_t)p[1] ||
                (p[3] == 0) != (p[1] == 0) || p[3] > 1)
                return false;
            rank = p[1], n_elem = 1;
            for (int i = 0; i < rank; ++i)
            {
                dims[i] = get_le(p + 4 + 8 * i, 8);
                if (!multiply(n_elem, dims[i], n_elem))
                    return false;
            }
            return true;
        }

        // Link Info message (version 0), compact storage; flags 0 or 3. Dense link storage throws "Unsupported".
        static inline bool decode_link_info(const uint8_t *p, size_t n, uint8_t &flags, uint64_t &max_corder)
        {
            if (n < 2 || p[0] != 0 || (p[1] != 0 && p[1] != 3) || n != (p[1] == 0 ? 18u : 34u))
                return false;
            flags = p[1];
            size_t pos = 2;
            max_corder = 0;
            if (flags != 0)
                max_corder = get_le(p + pos, 8), pos += 8;
            for (; pos < n; pos += 8)
                if (get_le(p + pos, 8) != undef)
                    unsupported();
            return true;
        }

        // Attribute message (version 3)
        static inline bool decode_attribute(const uint8_t *p, size_t n, attribute &a)
        {
            if (n < 9 || p[0] != 3 || p[1] != 0 || p[8] > 1)
                return false;
            const size_t name_size = (size_t)get_le(p + 2, 2), dt_size = (size_t)get_le(p + 4, 2),
                         ds_size = (size_t)get_le(p + 6, 2);
            if (name_size < 2 || name_size + dt_size + ds_size > n - 9 || p[9 + name_size - 1] != 0)
                return false;
            a.name = std::string_view((const char *)p + 9, name_size - 1);
            const uint8_t *dt = p + 9 + name_size, *ds = dt + dt_size;
            uint64_t value_size = 0;
            if (!decode_datatype(dt, dt_size, a.type) || !decode_dataspace(ds, ds_size, a.rank, a.dims, a.n_elem) ||
                !multiply(a.n_elem, a.type.size, value_size) || value_size != n - 9 - name_size - dt_size - ds_size)
                return false;
            a.value = ds + ds_size;
            return true;
        }

        // Decodes a group header: exactly one Link Info and one Group Info message, Link and Attribute messages, at
        // most one Attribute Info message (exactly one if there are attributes), nothing that belongs to a dataset
        static inline group decode_group(const object &o)
        {
            group g;
            auto bad = [&](const std::string &what)
            { malformed(what + " in the group at address " + std::to_string(o.chunks[0].addr)); };
            int n_link_info = 0, n_group_info = 0, n_attribute_info = 0;
            for (const chunk &c : o.chunks)
                for (const message_ref &m : c.messages)
                {
                    const uint8_t *p = c.data(m);
                    switch (m.type)
                    {
                    case msg_link:
                    {
                        link l;
                        if (!decode_link(p, m.size, l))
                            bad("invalid link message");
                        g.links.push_back(l);
                        break;
                    }
                    case msg_attribute:
                    {
                        attribute a;
                        if (!decode_attribute(p, m.size, a))
                            bad("invalid attribute message");
                        g.attributes.push_back(a);
                        break;
                    }
                    case msg_link_info:
                        ++n_link_info;
                        if (!decode_link_info(p, m.size, g.link_info_flags, g.max_corder))
                            bad("invalid link info message");
                        break;
                    case msg_group_info:
                        ++n_group_info;
                        if (m.size != 2 || p[0] != 0 || p[1] != 0)
                            bad("invalid group info message");
                        break;
                    case msg_attribute_info:
                        ++n_attribute_info;
                        if (m.size != 18 || p[0] != 0 || p[1] != 0)
                            bad("invalid attribute info message");
                        if (get_le(p + 2, 8) != undef || get_le(p + 10, 8) != undef)
                            unsupported(); // Dense attribute storage
                        break;
                    case msg_nil:
                    case msg_continuation:
                        break;
                    default:
                        malformed("the object at address " + std::to_string(o.chunks[0].addr) + " is not a group");
                    }
                }
            if (n_link_info != 1 || n_group_info != 1)
                malformed("the object at address " + std::to_string(o.chunks[0].addr) + " is not a group");
            if (n_attribute_info > 1 || (n_attribute_info == 0 && !g.attributes.empty()))
                bad("invalid attribute info");
            return g;
        }

        // Decodes a dataset header: exactly one Dataspace, Datatype and Data Layout message, at most one Fill Value
        // message; the raw data size must match the elements and lie within 'eof'
        static inline dataset decode_dataset(const object &o, uint64_t eof)
        {
            dataset d;
            auto bad = [&](const std::string &what)
            { malformed("the dataset at address " + std::to_string(o.chunks[0].addr) + " " + what); };
            int n_dataspace = 0, n_datatype = 0, n_layout = 0, n_fill = 0;
            uint64_t stored = 0;
            for (const chunk &c : o.chunks)
                for (const message_ref &m : c.messages)
                {
                    const uint8_t *p = c.data(m);
                    switch (m.type)
                    {
                    case msg_dataspace:
                        ++n_dataspace;
                        if (!decode_dataspace(p, m.size, d.rank, d.dims, d.n_elem))
                            bad("has an unsupported dataspace");
                        break;
                    case msg_datatype:
                        ++n_datatype;
                        if (!decode_datatype(p, m.size, d.type))
                            bad("has an unsupported datatype");
                        break;
                    case msg_fill_value:
                        ++n_fill;
                        if (m.size < 2 || p[0] != 3)
                            bad("has an unsupported fill value message");
                        break;
                    case msg_layout:
                        ++n_layout;
                        if (m.size >= 4 && p[0] == 3 && p[1] == 0 && m.size == 4 + get_le(p + 2, 2))
                            d.compact = true, d.data = p + 4, stored = get_le(p + 2, 2);
                        else if (m.size == 18 && p[0] == 3 && p[1] == 1)
                            d.compact = false, d.address = get_le(p + 2, 8), stored = get_le(p + 10, 8);
                        else
                            bad("has an unsupported data layout");
                        break;
                    case msg_nil:
                    case msg_continuation:
                        break;
                    default:
                        malformed("the object at address " + std::to_string(o.chunks[0].addr) + " is not a dataset");
                    }
                }
            if (n_dataspace != 1 || n_datatype != 1 || n_layout != 1 || n_fill > 1)
                malformed("the object at address " + std::to_string(o.chunks[0].addr) + " is not a dataset");
            if (!multiply(d.n_elem, d.type.size, d.size) || d.size != stored)
                bad("declares " + std::to_string(d.n_elem) + " elements but stores " +
                          std::to_string(stored) + " bytes");
            if (!d.compact && (d.size == 0 ? d.address != undef : !in_bounds(d.address, d.size, eof, superblock_size)))
                malformed("address " + std::to_string(d.address) + " is outside the file");
            return d;
        }

        // Reads the raw data of a dataset (size bytes) into 'dst'
        static inline void read_raw(file &f, const dataset &d, void *dst)
        {
            if (d.size == 0) // 'dst' may be null then (empty Armadillo object)
                return;
            if (d.compact)
                std::memcpy(dst, d.data, (size_t)d.size);
            else
                f.read(d.address, dst, (size_t)d.size);
        }

        // ---- Writing ----
        // Append-only output: bytes are collected in 'buf' (file address 'base' = address of buf[0]) and written when
        // the buffer is full; large raw data is written directly.
        class output
        {
        public:
            output(file &f, uint64_t addr) : base(addr), f_(f) { buf.reserve(io_block + max_chunk_body); }
            uint64_t base;
            std::vector<uint8_t> buf;

            uint64_t addr() const { return base + buf.size(); } // Address of the next byte
            void flush()
            {
                f_.write(base, buf.data(), buf.size());
                base += buf.size();
                buf.clear();
            }
            void flush_if_full()
            {
                if (buf.size() >= io_block)
                    flush();
            }

            // Appends n bytes; nullptr appends zeros
            void append(const void *src, uint64_t n)
            {
                if (src != nullptr && n >= io_block)
                {
                    flush();
                    f_.write(base, src, (size_t)n);
                    base += n;
                    return;
                }
                for (uint64_t done = 0; done < n; flush_if_full()) // Zeros in pieces of at most 1 MiB
                {
                    const size_t piece = (size_t)std::min(n - done, (uint64_t)io_block);
                    if (src != nullptr)
                        buf.insert(buf.end(), (const uint8_t *)src + done, (const uint8_t *)src + done + piece);
                    else
                        buf.resize(buf.size() + piece, 0);
                    done += piece;
                }
            }

            // Appends n values converted to float32
            void append_float(const double *src, uint64_t n)
            {
                for (uint64_t done = 0; done < n; flush_if_full())
                {
                    const size_t piece = (size_t)std::min(n - done, (uint64_t)io_block / 4);
                    const size_t pos = buf.size();
                    buf.resize(pos + 4 * piece);
                    for (size_t i = 0; i < piece; ++i)
                    {
                        const float value = (float)src[done + i];
                        std::memcpy(&buf[pos + 4 * i], &value, 4);
                    }
                    done += piece;
                }
            }

        private:
            file &f_;
        };

        // Appends a dataset: its header and, for contiguous storage, the raw data directly after it; returns the
        // header address. The dataset is compact if its raw data has at most 4096 bytes, unless 'force_contiguous'.
        // 'data' holds the 'n_bytes' raw bytes of the dataset (nullptr: zeros), or for a float32 dataset and
        // T = double, the n_bytes / 4 values to be converted.
        template <typename T>
        static inline uint64_t emit_dataset(output &out, dtype t, int rank, const uint64_t *dims, const T *data,
                                            uint64_t n_bytes, bool force_contiguous = false)
        {
            constexpr bool from_double = std::is_same<T, double>::value;
            const bool convert = from_double && t.t == type_t::f32;
            const bool compact = !force_contiguous && n_bytes <= compact_limit;
            const size_t body = (4 + 4 + 8 * (size_t)rank) + (4 + datatype_size(t)) + 6 +
                                (compact ? 4 + 4 + (size_t)n_bytes : 4 + 18);
            const size_t length = (body < 256 ? 7 : 8) + body + 4;
            const uint64_t addr = out.addr();
            std::vector<uint8_t> &b = out.buf;
            const size_t start = b.size();

            const uint8_t code = body < 256 ? 0 : 1;
            b.insert(b.end(), {'O', 'H', 'D', 'R', 2, code});
            put_le(b, body, 1 + code);
            dataspace_message(b, rank, dims);
            datatype_message(b, t);
            fill_value_message(b, compact);
            const size_t pos = begin_message(b, msg_layout);
            if (compact)
            {
                b.insert(b.end(), {3, 0});
                put_le(b, n_bytes, 2);
                const size_t at = b.size();
                b.resize(at + (size_t)n_bytes, 0);
                if constexpr (from_double)
                    if (convert)
                        for (size_t i = 0; i < (size_t)n_bytes / 4; ++i)
                        {
                            const float value = (float)data[i];
                            std::memcpy(&b[at + 4 * i], &value, 4);
                        }
                if (!convert && data != nullptr && n_bytes != 0)
                    std::memcpy(&b[at], (const void *)data, (size_t)n_bytes);
            }
            else
            {
                b.insert(b.end(), {3, 1});
                put_le(b, n_bytes == 0 ? undef : addr + length, 8);
                put_le(b, n_bytes, 8);
            }
            end_message(b, pos);
            put_le(b, lookup3(&b[start], b.size() - start), 4);

            if (!compact)
            {
                if constexpr (from_double)
                    if (convert)
                        out.append_float(data, n_bytes / 4);
                if (!convert)
                    out.append(data, n_bytes);
            }
            out.flush_if_full();
            return addr;
        }

        // Object header builder for groups: messages are added in order, 'emit' splits them into chunks
        class header
        {
        public:
            void clear() { msgs_.clear(), ends_.clear(), has_attributes_ = false; }

            void link_info(bool creation_order, uint64_t max_corder = 0)
            {
                const size_t pos = begin_message(msgs_, msg_link_info);
                msgs_.insert(msgs_.end(), {0, (uint8_t)(creation_order ? 3 : 0)});
                if (creation_order)
                    put_le(msgs_, max_corder, 8);
                for (int i = creation_order ? 3 : 2; i > 0; --i)
                    put_le(msgs_, undef, 8);
                end(pos);
            }

            void group_info()
            {
                const size_t pos = begin_message(msgs_, msg_group_info);
                msgs_.insert(msgs_.end(), {0, 0});
                end(pos);
            }

            // Attribute with a scalar (rank 0) or simple dataspace; 'value' holds n_elem * type size bytes. The
            // Attribute Info message is added before the first attribute.
            void attribute(std::string_view name, dtype t, int rank, const uint64_t *dims, const void *value)
            {
                if (!has_attributes_)
                {
                    const size_t pos = begin_message(msgs_, msg_attribute_info);
                    msgs_.insert(msgs_.end(), {0, 0});
                    put_le(msgs_, undef, 8);
                    put_le(msgs_, undef, 8);
                    end(pos);
                    has_attributes_ = true;
                }
                uint64_t n_elem = 1;
                for (int i = 0; i < rank; ++i)
                    n_elem *= dims[i];
                const size_t pos = begin_message(msgs_, msg_attribute);
                msgs_.insert(msgs_.end(), {3, 0});
                put_le(msgs_, name.size() + 1, 2);
                put_le(msgs_, datatype_size(t), 2);
                put_le(msgs_, 4 + 8 * (size_t)rank, 2);
                msgs_.push_back(0); // ASCII name
                msgs_.insert(msgs_.end(), name.begin(), name.end());
                msgs_.push_back(0);
                encode_datatype(msgs_, t);
                encode_dataspace(msgs_, rank, dims);
                const uint8_t *v = (const uint8_t *)value;
                msgs_.insert(msgs_.end(), v, v + n_elem * t.size);
                end(pos);
            }

            void link(std::string_view name, uint64_t target, uint64_t corder = undef)
            {
                link_message(msgs_, name, target, corder);
                ends_.push_back(msgs_.size());
            }

            // Bytes that emit() appends
            size_t size(size_t reserve) const
            {
                size_t n = 0;
                for (const auto &c : plan(reserve))
                    n += c.length;
                return n;
            }

            // Appends the header to 'block' (whose first byte has the file address 'base') and returns the address of
            // chunk 0. A message goes into the current chunk if the chunk body stays <= 65535 with 20 bytes left for a
            // Continuation message (with 'reserve' bytes after the last message); otherwise a Continuation message
            // closes the chunk and the next chunk follows directly. The last chunk keeps 'reserve' bytes of free space.
            uint64_t emit(std::vector<uint8_t> &block, uint64_t base, size_t reserve) const
            {
                const uint64_t addr = base + block.size();
                const std::vector<part> parts = plan(reserve);
                uint64_t next = addr;
                for (size_t k = 0; k < parts.size(); ++k)
                {
                    const size_t begin = parts[k].first == 0 ? 0 : ends_[parts[k].first - 1], end = ends_[parts[k].last - 1];
                    next += parts[k].length;
                    if (k + 1 == parts.size())
                        encode_chunk(block, k == 0, msgs_.data() + begin, end - begin, parts[k].body);
                    else
                    {
                        std::vector<uint8_t> msgs(msgs_.begin() + begin, msgs_.begin() + end);
                        continuation_message(msgs, next, parts[k + 1].length);
                        encode_chunk(block, k == 0, msgs.data(), msgs.size(), parts[k].body);
                    }
                }
                return addr;
            }

        private:
            struct part
            {
                size_t first, last; // Messages [first, last)
                size_t body, length;
            };

            std::vector<part> plan(size_t reserve) const
            {
                std::vector<part> parts;
                size_t first = 0, used = 0;
                for (size_t i = 0; i < ends_.size(); ++i)
                {
                    const size_t size = ends_[i] - (i == 0 ? 0 : ends_[i - 1]);
                    const size_t room = (i + 1 == ends_.size()) ? reserve : continuation_size;
                    if (used != 0 && used + size + room > max_chunk_body)
                        parts.push_back({first, i, used + continuation_size, 0}), first = i, used = 0;
                    used += size;
                }
                parts.push_back({first, ends_.size(), used + reserve, 0});
                for (size_t k = 0; k < parts.size(); ++k)
                    parts[k].length = (k == 0 ? (parts[k].body < 256 ? 7 : 8) : 4) + parts[k].body + 4;
                return parts;
            }

            void end(size_t pos)
            {
                end_message(msgs_, pos);
                ends_.push_back(msgs_.size());
            }

            std::vector<uint8_t> msgs_; // Encoded messages
            std::vector<size_t> ends_;  // End of each message in msgs_
            bool has_attributes_ = false;
        };

        // ---- Growable group headers ----
        // A group header whose last chunk has free space for new links (channel groups: room for exactly one link;
        // groups with creation order: room for a Continuation message). The free space of the last chunk is computed
        // from its messages, and a rewrite compacts the chunk (non-NIL messages in order, the new message, the tail).
        class chain_links
        {
        public:
            void load(file &f, uint64_t addr, uint64_t eof)
            {
                obj_ = read_object(f, addr, eof);
                grp_ = decode_group(obj_);
            }
            const link *find(std::string_view name) const { return grp_.find(name); }
            std::vector<std::pair<uint64_t, uint64_t>> ranges() const { return chunk_ranges(obj_); }

            // Throws the "Unsupported" error if a link with a name of 'name_length' bytes cannot be inserted. Groups
            // with creation order ('creation_order') need Link Info flags 3 and grow by new chunks; the others must
            // have room for the link in their last chunk.
            void check_insert(size_t name_length, bool creation_order) const
            {
                const size_t L = link_message_size(name_length, creation_order), F = obj_.chunks.back().free_space;
                if (creation_order ? (grp_.link_info_flags != 3 || grp_.max_corder >= (1ULL << 63) || F < continuation_size)
                                   : L > F)
                    unsupported();
            }

            // Inserts a link; a new continuation chunk, if needed, is appended to 'block' (first byte at address
            // 'base'). Returns the chunk rewrites in the order to be written: the chunk holding the Link Info message
            // (new creation index), then the last chunk. Call check_insert first; the views become invalid.
            std::vector<patch> prepare_insert(std::string_view name, uint64_t target, bool creation_order,
                                              std::vector<uint8_t> &block, uint64_t base)
            {
                check_insert(name.size(), creation_order);
                std::vector<patch> patches;
                std::vector<uint8_t> link_msg;
                link_message(link_msg, name, target, creation_order ? grp_.max_corder : undef);
                chunk *info_chunk = nullptr;
                if (creation_order) // Creation index + 1, in the image of the chunk holding the Link Info message
                    for (chunk &c : obj_.chunks)
                        for (const message_ref &m : c.messages)
                            if (m.type == msg_link_info)
                                put_le(&c.image[m.offset + 2], grp_.max_corder + 1, 8), info_chunk = &c;

                chunk &last = obj_.chunks.back();
                std::vector<uint8_t> msgs = chunk_messages(last);
                if (!creation_order || last.free_space >= link_msg.size() + continuation_size)
                    msgs.insert(msgs.end(), link_msg.begin(), link_msg.end());
                else // New chunk with the link and free space, linked from the last chunk
                {
                    const size_t body = std::max(link_msg.size() + continuation_size,
                                                 std::min(chain_chunk_body_max, 2 * last.body()));
                    continuation_message(msgs, base + block.size(), body + 8);
                    encode_chunk(block, false, link_msg.data(), link_msg.size(), body);
                }
                if (info_chunk != nullptr && info_chunk != &last)
                {
                    rewrite_chunk(*info_chunk, chunk_messages(*info_chunk));
                    patches.push_back({info_chunk->addr, info_chunk->image});
                }
                rewrite_chunk(last, msgs);
                patches.push_back({last.addr, last.image});
                return patches;
            }

        private:
            object obj_;
            group grp_;
        };

        // ---- Root group links ----
        // Root chunk 0 holds Link Info, Group Info and the fixed links written at creation, then the directory:
        // Continuation messages to link chunks and, as the last entry of a full directory chunk, to the next directory
        // chunk. Link chunks hold links named <prefix><id> (id = 1 ... 2^32 - 1, increasing) and the tail. Directory
        // and link chunks lie in the file in creation order. A writer reads chunk 0, all directory chunks, the last
        // link chunk and, for a removal, the link chunk holding the old link; the order of all other entries is
        // checked, their chunks are not read. Every violation of this structure throws the "Unsupported" error.
        class root_links
        {
        public:
            explicit root_links(std::string prefix) : prefix_(std::move(prefix)) {}

            // Checks root chunk 0: Link Info (no creation order), Group Info, 'n_fixed' links, then only Continuation
            // messages and the tail, with free space 0 or a multiple of 20
            static void check_chunk0(const chunk &c0, size_t n_fixed)
            {
                const auto &m = c0.messages;
                if (m.size() < 2 + n_fixed || m[0].type != msg_link_info || m[1].type != msg_group_info ||
                    m[0].size != 18 || c0.data(m[0])[1] != 0)
                    unsupported();
                for (size_t i = 2; i < m.size(); ++i)
                    if (m[i].type != (i < 2 + n_fixed ? msg_link : msg_continuation) &&
                        !(m[i].type == msg_nil && i + 1 == m.size()))
                        unsupported();
                if (c0.free_space % continuation_size != 0)
                    unsupported();
            }

            // Reads the directory chunks, the last link chunk and, if 'old_target' is not 0, the link chunk holding the
            // link to 'old_target'
            void load(file &f, const chunk &c0, uint64_t eof, uint64_t old_target)
            {
                dirs_.assign(1, c0);
                uint64_t next = c0.addr + c0.length();
                for (size_t d = 0; d < dirs_.size(); ++d)
                {
                    if (d != 0) // Directory chunks: only Continuation messages and the tail
                        for (size_t i = 0; i < dirs_[d].messages.size(); ++i)
                            if (dirs_[d].messages[i].type != msg_continuation &&
                                !(dirs_[d].messages[i].type == msg_nil && i + 1 == dirs_[d].messages.size()))
                                unsupported();
                    if (dirs_[d].free_space % continuation_size != 0)
                        unsupported();
                    std::vector<entry> entries;
                    for (const message_ref &m : dirs_[d].messages)
                        if (m.type == msg_continuation)
                            entries.push_back({get_le(dirs_[d].data(m), 8), get_le(dirs_[d].data(m) + 8, 8)});
                    const bool full = dirs_[d].free_space == 0;
                    for (size_t i = 0; i < entries.size(); ++i)
                    {
                        const entry e = entries[i];
                        if (e.length < 8 || e.length > max_chunk_body + 8 || !in_bounds(e.addr, e.length, eof, superblock_size))
                            malformed("address " + std::to_string(e.addr) + " is outside the file");
                        if (e.addr < next)
                            malformed("continuation chunk at address " + std::to_string(e.addr) + " is out of order");
                        next = e.addr + e.length;
                        if (full && i + 1 == entries.size())
                            dirs_.push_back(read_chunk(f, e.addr, e.length, eof));
                        else
                            links_.push_back(e);
                    }
                }
                if (dirs_.back().free_space < continuation_size)
                    unsupported();
                last_id_ = 0;
                if (!links_.empty())
                {
                    load_link_chunk(f, links_.size() - 1, eof);
                    const chunk &last = loaded_.back();
                    if (last.messages.empty() || last.messages[0].type != msg_link)
                        unsupported();
                    size_t i = last.messages.size() - (last.messages.back().type == msg_nil ? 2 : 1);
                    link l;
                    decode_link(last.data(last.messages[i]), last.messages[i].size, l);
                    last_id_ = (uint64_t)parse_decimal(l.name.substr(prefix_.size()));
                }
                const size_t k = find(old_target);
                if (old_target != 0 && k != npos && k + 1 != links_.size())
                    load_link_chunk(f, k, eof);
            }

            // ID of the last link (the largest ID ever linked); 0 if there is no link chunk
            uint64_t last_id() const { return last_id_; }

            // Address and length of every chunk read by 'load' (the chunks that prepare_insert/prepare_remove rewrite)
            std::vector<std::pair<uint64_t, uint64_t>> ranges() const
            {
                std::vector<std::pair<uint64_t, uint64_t>> r;
                for (const chunk &c : dirs_)
                    r.push_back({c.addr, c.length()});
                for (const chunk &c : loaded_)
                    r.push_back({c.addr, c.length()});
                return r;
            }

            // Bytes that prepare_insert appends to the block for a link with a name of 'name_length' bytes
            size_t insert_size(size_t name_length) const
            {
                if (!links_.empty() && link_message_size(name_length, false) <= loaded_link_chunk(links_.size() - 1)->free_space)
                    return 0;
                return 8 + link_chunk_body(links_.size()) +
                       (dirs_.back().free_space == continuation_size ? 8 + directory_chunk_body : 0);
            }

            // Appends the link to the last link chunk, or creates a new link chunk (and, if the last directory chunk
            // has room for one entry only, a new directory chunk first) in 'block' (first byte at address 'base').
            // Returns the one chunk rewrite: the last link chunk or the last directory chunk.
            patch prepare_insert(std::string_view name, uint64_t target, std::vector<uint8_t> &block, uint64_t base)
            {
                std::vector<uint8_t> link_msg;
                link_message(link_msg, name, target);
                last_id_ = (uint64_t)parse_decimal(name.substr(prefix_.size()));
                if (!links_.empty())
                {
                    chunk &last = *loaded_link_chunk(links_.size() - 1);
                    if (link_msg.size() <= last.free_space)
                    {
                        std::vector<uint8_t> msgs = chunk_messages(last);
                        msgs.insert(msgs.end(), link_msg.begin(), link_msg.end());
                        rewrite_chunk(last, msgs);
                        return {last.addr, last.image};
                    }
                }
                const size_t body = link_chunk_body(links_.size());
                chunk &dir = dirs_.back();
                std::vector<uint8_t> dir_msgs = chunk_messages(dir);
                if (dir.free_space == continuation_size) // New directory chunk, then the link chunk
                {
                    const uint64_t dir_addr = base + block.size(), link_addr = dir_addr + 8 + directory_chunk_body;
                    continuation_message(dir_msgs, dir_addr, 8 + directory_chunk_body);
                    std::vector<uint8_t> first_entry;
                    continuation_message(first_entry, link_addr, 8 + body);
                    chunk new_dir;
                    new_dir.addr = dir_addr, new_dir.ohdr = false;
                    encode_chunk(new_dir.image, false, first_entry.data(), first_entry.size(), directory_chunk_body);
                    parse_chunk(new_dir);
                    block.insert(block.end(), new_dir.image.begin(), new_dir.image.end());
                    rewrite_chunk(dir, dir_msgs);
                    patch p{dir.addr, dir.image};
                    dirs_.push_back(std::move(new_dir));
                    append_link_chunk(link_msg, body, block, base);
                    return p;
                }
                continuation_message(dir_msgs, base + block.size(), 8 + body);
                rewrite_chunk(dir, dir_msgs);
                append_link_chunk(link_msg, body, block, base);
                return {dir.addr, dir.image};
            }

            // Removes the link to 'target' from the link chunk with the largest address below 'target' (loaded by
            // 'load'); false if there is no such link
            bool prepare_remove(uint64_t target, patch &p)
            {
                const size_t k = find(target);
                chunk *c = k == npos ? nullptr : loaded_link_chunk(k);
                if (c == nullptr)
                    return false;
                for (const message_ref &m : c->messages)
                {
                    link l;
                    if (m.type == msg_link && decode_link(c->data(m), m.size, l) && l.target == target)
                    {
                        rewrite_chunk(*c, chunk_messages(*c, &m));
                        p = {c->addr, c->image};
                        return true;
                    }
                }
                return false;
            }

        private:
            struct entry
            {
                uint64_t addr, length;
            };
            static constexpr size_t npos = ~(size_t)0;

            static size_t link_chunk_body(size_t k)
            {
                return k >= 3 ? link_chunk_body_max : std::min(link_chunk_body_max, link_chunk_body_first << k);
            }

            // Index of the link chunk with the largest address below 'target'; npos if none
            size_t find(uint64_t target) const
            {
                auto it = std::lower_bound(links_.begin(), links_.end(), target, [](const entry &e, uint64_t t)
                                           { return e.addr < t; });
                return it == links_.begin() ? npos : (size_t)(it - links_.begin()) - 1;
            }

            chunk *loaded_link_chunk(size_t k)
            {
                for (chunk &c : loaded_)
                    if (c.addr == links_[k].addr)
                        return &c;
                return nullptr;
            }
            const chunk *loaded_link_chunk(size_t k) const { return const_cast<root_links *>(this)->loaded_link_chunk(k); }

            // Reads link chunk k and checks it: only hard links named <prefix><id> with increasing IDs, and the tail
            void load_link_chunk(file &f, size_t k, uint64_t eof)
            {
                chunk c = read_chunk(f, links_[k].addr, links_[k].length, eof);
                long long previous = 0;
                for (size_t i = 0; i < c.messages.size(); ++i)
                {
                    const message_ref &m = c.messages[i];
                    if (m.type == msg_nil && i + 1 == c.messages.size())
                        break;
                    link l;
                    if (m.type != msg_link || !decode_link(c.data(m), m.size, l) || l.name.substr(0, prefix_.size()) != prefix_)
                        unsupported();
                    const long long id = parse_decimal(l.name.substr(prefix_.size()));
                    if (id <= previous)
                        unsupported();
                    previous = id;
                }
                loaded_.push_back(std::move(c));
            }

            void append_link_chunk(const std::vector<uint8_t> &link_msg, size_t body, std::vector<uint8_t> &block, uint64_t base)
            {
                chunk c;
                c.addr = base + block.size(), c.ohdr = false;
                encode_chunk(c.image, false, link_msg.data(), link_msg.size(), body);
                parse_chunk(c);
                block.insert(block.end(), c.image.begin(), c.image.end());
                links_.push_back({c.addr, c.length()});
                loaded_.push_back(std::move(c));
            }

            std::string prefix_;
            std::vector<chunk> dirs_;   // Root chunk 0 and the directory chunks; only the last one changes
            std::vector<entry> links_;  // Link-chunk entries in reading order (increasing addresses)
            std::vector<chunk> loaded_; // Link chunks read or created
            uint64_t last_id_ = 0;
        };
    }
}

#endif
