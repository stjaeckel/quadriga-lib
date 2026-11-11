#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
#
# Checks that the AVX2 object files of a library build contain AVX code only in the AVX2 kernels.
#
# The files src/*_avx2.cpp are compiled with -mavx2 -mfma, so every function emitted there may use AVX instructions
# (and popcnt, which -mavx2 also enables). Templates and inline functions (std::vector, std::unique_ptr, ...) are
# emitted as weak symbols, and the linker keeps one copy for all callers. If it keeps the AVX2 copy, plain code calls
# it and stops with "Illegal instruction" on a CPU without AVX2. The kernels are safe because they run only after
# runtime_AVX2_Check(). Other global functions, such as runtime_AVX2_Check() itself, and static initializers must run
# on every CPU.
#
# Usage: bash tests/check_avx2_objects.sh <cmake build folder>
# Fails (exit 1) when a global, weak or startup function that is not a kernel (name ending in _AVX2 or _avx2)
# contains AVX, popcnt or crc32 instructions. Exits with 2 when the objects cannot be read. Needs nm, objdump and
# c++filt (GNU binutils or LLVM). Not for MSVC builds.

set -u

fail()
{
    echo "check_avx2_objects: $*" >&2
    exit 2
}

if [ $# -ne 1 ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <cmake build folder>" >&2
    exit 2
fi
[ -f "$1/CMakeCache.txt" ] || fail "'$1' is not a CMake build folder (no CMakeCache.txt)"

for tool in nm objdump c++filt; do
    command -v "$tool" >/dev/null 2>&1 || fail "'$tool' not found (GNU binutils or LLVM needed)"
done

# Objects of the current AVX2 sources only; a build folder can keep objects of renamed or removed files
src_dir="$(cd "$(dirname "$0")/.." && pwd)/src"
objects=()
for src in "$src_dir"/*_avx2.cpp; do
    [ -e "$src" ] || continue
    while IFS= read -r obj; do
        objects+=("$obj")
    done < <(find "$1" -path '*CMakeFiles/quadriga_*.dir/*' -name "$(basename "$src").o")
done

if [ ${#objects[@]} -eq 0 ]; then
    echo "check_avx2_objects: no AVX2 objects in '$1' (AVX2 disabled or not x86-64), nothing to check"
    exit 0
fi

# Kernels: a function whose own name ends in _AVX2 or _avx2, with or without template arguments or clone suffix
kernel_re='(^|[^:A-Za-z0-9_])[A-Za-z0-9_]+_(AVX2|avx2)(<[^()]*>)?\('

n_kernel=0 # Instructions found in the kernels; none means that the disassembly was not understood
found=""
for obj in "${objects[@]}"; do
    syms=$(nm --defined-only "$obj") || fail "nm cannot read '$obj'"
    dis=$(objdump -d --no-show-raw-insn "$obj") || fail "objdump cannot read '$obj'"

    # Functions that other objects can call: global (T), weak (W) and static initializers
    checked=$(awk '$2 == "T" || $2 == "W" || $3 ~ /^_GLOBAL__sub_I_/ { print $3 }' <<< "$syms" | sort -u)

    # Number of AVX (mnemonic v...), popcnt and crc32 instructions per function
    # GNU objdump puts a tab after the address, LLVM objdump spaces and a tab
    counts=$(awk '
        /^[0-9a-f]+ <.*>:$/ { name = substr($2, 2, length($2) - 3); next }
        /^ *[0-9a-f]+:[ \t]+(v[a-z0-9]+|popcnt[a-z]*|crc32[a-z]*)([ \t]|$)/ { if (name != "") n[name]++ }
        END { for (s in n) print s, n[s] }' <<< "$dis")

    while read -r sym count; do
        [ -z "$sym" ] && continue
        name=$(c++filt -- "$sym")
        if grep -qE "$kernel_re" <<< "$name"; then
            n_kernel=$((n_kernel + count))
            continue
        fi
        grep -qxF -- "$sym" <<< "$checked" || continue
        found+="  $(basename "$obj"): $count instructions in $name"$'\n'
    done <<< "$counts"
done

[ "$n_kernel" -gt 0 ] || fail "no AVX instructions found in the kernels, the output of objdump was not understood"

if [ -n "$found" ]; then
    echo "check_avx2_objects: functions with AVX code outside the AVX2 kernels:"
    printf '%s' "$found"
    echo "The linker may use these copies for plain code, which then crashes on CPUs without AVX2."
    echo "Keep std containers, smart pointers and other templates out of the src/*_avx2.cpp files."
    exit 1
fi

echo "check_avx2_objects: ${#objects[@]} AVX2 objects checked, AVX code only in the kernels"
exit 0
