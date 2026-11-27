// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Public types, constants and small inline helpers that more than one public header needs. Includes only Armadillo
// and the standard library; declarations of library functions belong to the topic headers.

#ifndef quadriga_common_H
#define quadriga_common_H

#include <armadillo>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace quadriga_lib
{
    // List of non-negative integers for index and count arguments. Takes a uvec (used in place, no copy), a single
    // value, a brace list, a std::vector, or an Armadillo vector or expression of any integer or floating-point type;
    // a negative, fractional or non-finite value throws std::invalid_argument. Meant for function arguments only: it
    // may refer to the uvec it was given.
    class uint_list
    {
        template <typename T> // Integer and floating-point types, not bool
        using number_only = typename std::enable_if<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, int>::type;
        template <typename... T> // One or more numbers
        using numbers_only = typename std::enable_if<(sizeof...(T) > 0) && ((std::is_arithmetic<T>::value && !std::is_same<T, bool>::value) && ...), int>::type;

    public:
        uint_list() {}
        uint_list(const arma::uvec &v) : ref(&v) {}
        uint_list(arma::uvec &&v) : own(std::move(v)) {}
        uint_list(const arma::uvec &&v) : own(v) {} // A const temporary is copied, not referenced
        template <typename... T, numbers_only<T...> = 0> // A single value or a brace list of mixed types, e.g. {n, 2}
        uint_list(T... i) : own{cast(i)...} {}
        template <typename T, number_only<T> = 0>
        uint_list(const std::vector<T> &v) { fill(v.begin(), v.size()); }
        template <typename T, typename T1, number_only<T> = 0> // Armadillo vectors and expressions, e.g. find(x > 0)
        uint_list(const arma::Base<T, T1> &x)
        {
            const arma::Mat<T> m(x.get_ref());
            if (!m.is_vec() && !m.is_empty())
                throw std::invalid_argument("Index and count arguments must be vectors.");
            fill(m.begin(), m.n_elem);
        }

        const arma::uvec &vec() const { return ref ? *ref : own; } // The values as uvec

    private:
        arma::uvec own;
        const arma::uvec *ref = nullptr;
        template <typename T>
        static arma::uword cast(T v)
        {
            bool ok = true;
            if constexpr (std::is_floating_point<T>::value) // NaN fails the range test; at most 2^53
                ok = v >= T(0) && v <= T(9007199254740992.0) && std::floor(v) == v;
            else if constexpr (std::is_signed<T>::value)
                ok = v >= T(0);
            if (!ok)
                throw std::invalid_argument("Index and count arguments must be non-negative integers.");
            return (arma::uword)v;
        }
        template <typename It>
        void fill(It it, size_t n)
        {
            own.set_size((arma::uword)n);
            for (arma::uword k = 0; k < (arma::uword)n; ++k, ++it)
                own[k] = cast(*it);
        }
    };

    // List of real numbers for parameter tuples, frequency lists and other values a caller types by hand. Takes a vec
    // (used in place, no copy), a single value, a brace list, a std::vector, or an Armadillo vector or expression of
    // any integer or floating-point type; all values are kept, NaN and Inf included. Meant for function arguments
    // only: it may refer to the vec it was given.
    class double_list
    {
        template <typename T> // Integer and floating-point types, not bool
        using number_only = typename std::enable_if<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, int>::type;
        template <typename... T> // One or more numbers
        using numbers_only = typename std::enable_if<(sizeof...(T) > 0) && ((std::is_arithmetic<T>::value && !std::is_same<T, bool>::value) && ...), int>::type;

    public:
        double_list() {}
        double_list(const arma::vec &v) : ref(&v) {}
        double_list(arma::vec &&v) : own(std::move(v)) {}
        double_list(const arma::vec &&v) : own(v) {} // A const temporary is copied, not referenced
        template <typename... T, numbers_only<T...> = 0> // A single value or a brace list of mixed types, e.g. {n, 2.5}
        double_list(T... x) : own{double(x)...} {}
        template <typename T, number_only<T> = 0>
        double_list(const std::vector<T> &v) { fill(v.begin(), v.size()); }
        template <typename T, typename T1, number_only<T> = 0> // Armadillo vectors and expressions, e.g. linspace(...)
        double_list(const arma::Base<T, T1> &x)
        {
            const arma::Mat<T> m(x.get_ref());
            if (!m.is_vec() && !m.is_empty())
                throw std::invalid_argument("List arguments must be vectors.");
            fill(m.begin(), m.n_elem);
        }

        const arma::vec &vec() const { return ref ? *ref : own; } // The values as vec
        template <typename T>
        arma::Col<T> as() const { return arma::conv_to<arma::Col<T>>::from(vec()); } // A copy as Col<T>, e.g. as<dtype>()

    private:
        arma::vec own;
        const arma::vec *ref = nullptr;
        template <typename It>
        void fill(It it, size_t n)
        {
            own.set_size((arma::uword)n);
            for (arma::uword k = 0; k < (arma::uword)n; ++k, ++it)
                own[k] = double(*it);
        }
    };
}

#endif
