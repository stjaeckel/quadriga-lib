// SPDX-License-Identifier: Apache-2.0
// Provide std::__throw_bad_array_new_length locally for MATLAB installs that
// ship libstdc++ < GLIBCXX_3.4.29 (i.e. MATLAB versions bundled before GCC 11).
// GCC 11+ emits this symbol for any `new T[n]` with runtime n, including ones
// inside libstdc++ templates such as std::unordered_map / std::unordered_set.
//
// Only needed for libstdc++; libc++ and MSVC STL don't have this symbol.

#include <new>   // must precede the __GLIBCXX__ check — it's what defines the macro

#if defined(__GLIBCXX__)
namespace std {
    [[noreturn]] void __throw_bad_array_new_length() {
        throw std::bad_array_new_length();
    }
}
#endif