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

// How to use
//
// bits<T> wraps a value of a trivially copyable type T (sizeof 1, 2, 4, or 8)
// and lets you read and write its individual bits. T may be any integer type
// or float/double; bit math runs on an unsigned integer of the same width, so
// floating-point bit access is well defined.
//
// Construction:
//   qd::bits<uint32_t> a;          // zero-initialized
//   qd::bits<uint32_t> b = 0x0F;   // from a value
//   qd::bits<float>    f = 1.5f;   // from a float
//
// Bit order: index 0 is the least significant bit (value 1), index
// bit_count-1 is the most significant. a[0]=true yields value 1, a[7]=true
// yields 128. Caller must keep n < bit_count; shifting out of range is UB.
//
// Named operations (chainable, return *this):
//   a.set(3).clear(0).flip(7);     // single bits
//   a.assign(2, true);             // conditional set/clear
//   a.set(); a.clear(); a.flip();  // whole value
//   bool on = a.test(5);           // read a bit
//
// Indexed access:
//   a[3] = true;                   // write via proxy
//   a[7] = a[3];                   // copy one bit to another
//   bool on = a[5];                // read
//
// Float example (IEEE-754: bit 31 sign, 23-30 exponent, 0-22 mantissa):
//   qd::bits<float> f = 1.5f;
//   f.flip(31);                    // -> -1.5f
//   bool sign = f[31];
//
// Output / access:
//   uint32_t raw = a;              // implicit conversion to T
//   uint32_t raw = a.get();        // explicit getter
//   auto w = a.word();             // value reinterpreted as unsigned integer
//
// Reinterpreting an existing object: prefer copying via bit_cast over aliasing.
//   float x = 1.5f;
//   auto bx = std::bit_cast<qd::bits<float>>(x);  // C++20; defined and cheap
//   bx.flip(31);
//   x = bx.get();
// reinterpret_cast<bits<float>&>(x) on a live float works in practice but is
// undefined under strict aliasing; use bit_cast, or declare the variable as
// bits<float> from the start.
//
// Storage: bits<T> holds exactly one T, is standard-layout and trivially
// copyable, sizeof == sizeof(T), so it drops into arrays, structs, memcpy,
// and device code anywhere a bare T would.
//
// C++17: when std::bit_cast is unavailable the header falls back to a memcpy
// copy, which is not constant-evaluable, so compile-time use needs C++20.

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

        // Proxy for a single addressable bit.
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

        // Returns value with only the low n bits kept; higher bits zeroed.
        constexpr T tail(unsigned n) const
        {
            Word mask = (n >= bit_count) ? static_cast<Word>(~Word{0})
                                         : static_cast<Word>((Word{1} << n) - 1);
            return detail::bit_cast<T>(static_cast<Word>(word() & mask));
        }

        // Returns value with only the high n bits kept; lower bits zeroed
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

        // Whole-word operations.
        constexpr bits &set() { return apply(static_cast<Word>(~Word{0})); }
        constexpr bits &clear() { return apply(Word{0}); }
        constexpr bits &flip() { return apply(static_cast<Word>(~word())); }

        // Indexing. Non-const returns a writable proxy, const returns a plain bool.
        constexpr BitRef operator[](unsigned n) { return BitRef{value, bit(n)}; }
        constexpr bool operator[](unsigned n) const { return test(n); }

        // Access.
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