// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef bits_struct_H
#define bits_struct_H

#include <cstdint>
#include <cstddef>
#include <type_traits>

#if __has_include(<version>)
#include <version>
#endif

#if defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
#include <bit>
#else
#include <cstring>
#endif

// qd::bits<T> holds one value of a trivially copyable T (1, 2, 4 or 8 bytes, integer or float/double) and reads and
// writes its bits through an unsigned integer of the same width, so bit access to floats is well defined. Index 0 is
// the least significant bit; n must be < bit_count (larger shifts are UB). set/clear/flip/assign are chainable,
// a[n] returns a writable proxy, word() the raw unsigned value. Example (IEEE-754 float, bit 31 = sign):
//   qd::bits<float> f = 1.5f;
//   f.flip(31); // -> -1.5f
// sizeof(bits<T>) == sizeof(T), standard-layout and trivially copyable. To reinterpret an existing float, copy it with
// bit_cast; reinterpret_cast<bits<float>&>(x) violates strict aliasing. Without std::bit_cast (C++17) the header falls
// back to memcpy, which is not constant-evaluable.

namespace qd
{
    namespace detail
    {
#if defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
        using std::bit_cast;
#else
        template <class To, class From>
        To bit_cast(const From &src) noexcept
        {
            static_assert(sizeof(To) == sizeof(From), "bit_cast requires equal sizes");
            static_assert(std::is_trivially_copyable_v<To> && std::is_trivially_copyable_v<From>,
                          "bit_cast requires trivially copyable types");
            To dst;
            std::memcpy(&dst, &src, sizeof(To));
            return dst;
        }
#endif
    }

    template <class T>
    struct bits
    {
        static_assert(std::is_trivially_copyable_v<T>, "bits<T> requires a trivially copyable T");

        using Word = std::conditional_t<sizeof(T) == 1, std::uint8_t,
                                        std::conditional_t<sizeof(T) == 2, std::uint16_t,
                                                           std::conditional_t<sizeof(T) == 4, std::uint32_t,
                                                                              std::conditional_t<sizeof(T) == 8, std::uint64_t, void>>>>;

        static_assert(!std::is_void_v<Word>, "bits<T>: only 1, 2, 4, or 8-byte types are supported");
        static constexpr unsigned bit_count = sizeof(T) * 8;
        T value{};

        // Proxy for a single addressable bit
        struct BitRef
        {
            T &ref;
            Word mask;

            constexpr operator bool() const
            {
                return detail::bit_cast<Word>(ref) & mask;
            }
            constexpr BitRef &operator=(bool v)
            {
                Word w = detail::bit_cast<Word>(ref);
                w = v ? static_cast<Word>(w | mask) : static_cast<Word>(w & ~mask);
                ref = detail::bit_cast<T>(w);
                return *this;
            }
            constexpr BitRef &operator=(const BitRef &other)
            {
                return *this = static_cast<bool>(other);
            }
            constexpr BitRef &flip()
            {
                ref = detail::bit_cast<T>(static_cast<Word>(detail::bit_cast<Word>(ref) ^ mask));
                return *this;
            }
        };

        constexpr bits() = default;
        constexpr bits(T v) : value(v) {}

        constexpr Word word() const { return detail::bit_cast<Word>(value); }

        // Named single-bit operations, chainable. Caller ensures n < bit_count.
        constexpr bits &set(unsigned n) { return apply(static_cast<Word>(word() | bit(n))); }
        constexpr bits &clear(unsigned n) { return apply(static_cast<Word>(word() & ~bit(n))); }
        constexpr bits &flip(unsigned n) { return apply(static_cast<Word>(word() ^ bit(n))); }
        constexpr bits &assign(unsigned n, bool v) { return v ? set(n) : clear(n); }

        constexpr bool test(unsigned n) const { return word() & bit(n); }

        // Value with only the low n bits kept
        constexpr T tail(unsigned n) const
        {
            Word mask = (n >= bit_count) ? static_cast<Word>(~Word{0})
                                         : static_cast<Word>((Word{1} << n) - 1);
            return detail::bit_cast<T>(static_cast<Word>(word() & mask));
        }

        // Value with only the high n bits kept
        constexpr T head(unsigned n) const
        {
            Word mask;
            if (n == 0)
                mask = Word{0};
            else if (n >= bit_count)
                mask = static_cast<Word>(~Word{0});
            else
                mask = static_cast<Word>(~((Word{1} << (bit_count - n)) - 1));
            return detail::bit_cast<T>(static_cast<Word>(word() & mask));
        }

        // Whole-word operations
        constexpr bits &set() { return apply(static_cast<Word>(~Word{0})); }
        constexpr bits &clear() { return apply(Word{0}); }
        constexpr bits &flip() { return apply(static_cast<Word>(~word())); }

        // Non-const returns a writable proxy, const a plain bool
        constexpr BitRef operator[](unsigned n) { return BitRef{value, bit(n)}; }
        constexpr bool operator[](unsigned n) const { return test(n); }

        constexpr operator T() const { return value; }
        constexpr T get() const { return value; }

    private:
        static constexpr Word bit(unsigned n) { return static_cast<Word>(Word{1} << n); }
        constexpr bits &apply(Word w)
        {
            value = detail::bit_cast<T>(w);
            return *this;
        }
    };
}

#endif
