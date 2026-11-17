#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
#
# Checks that the MATLAB MEX files of a Linux build load with the C++ runtime of the oldest supported MATLAB release.
#
# MATLAB loads MEX files with its own bundled libstdc++ and libgcc_s, not with the system ones. A MEX file that needs
# a symbol version newer than the bundled libraries provide fails to load with "version `...' not found". Newer GCC
# releases emit such symbols for some standard library features (e.g. std::exception_ptr needs CXXABI_1.3.13 with
# GCC 11 and newer), so a build can pass all tests with a recent MATLAB release and still fail with an older one.
#
# The limits are those of MATLAB R2024a (libstdc++ and libgcc_s of GCC 10). The check reads only the dynamic symbol
# table, so it works for builds on any architecture.
#
# Usage: bash tests/check_mex_symbol_versions.sh <cmake build folder>
# Fails (exit 1) when a .mexa64 file in <build folder>/+quadriga_lib needs a GLIBCXX, CXXABI or GCC symbol version
# above the limits. Exits with 2 when the files cannot be read. Needs objdump and c++filt (GNU binutils or LLVM).

set -u

# Newest symbol versions in the libraries bundled with MATLAB R2024a (sys/os/glnxa64)
limits="GLIBCXX 3.4.28
CXXABI 1.3.12
GCC 7.0.0"

fail()
{
    echo "check_mex_symbol_versions: $*" >&2
    exit 2
}

if [ $# -ne 1 ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <cmake build folder>" >&2
    exit 2
fi
[ -f "$1/CMakeCache.txt" ] || fail "'$1' is not a CMake build folder (no CMakeCache.txt)"

for tool in objdump c++filt; do
    command -v "$tool" >/dev/null 2>&1 || fail "'$tool' not found (GNU binutils or LLVM needed)"
done

mex_files=()
for f in "$1"/+quadriga_lib/*.mexa64; do
    [ -e "$f" ] && mex_files+=("$f")
done

if [ ${#mex_files[@]} -eq 0 ]; then
    echo "check_mex_symbol_versions: no MATLAB MEX files in '$1' (MATLAB disabled), nothing to check"
    exit 0
fi

n_versioned=0 # References to libstdc++ symbol versions; none means that the output of objdump was not understood
found=""
for f in "${mex_files[@]}"; do
    syms=$(objdump -T "$f") || fail "objdump cannot read '$f'"

    # Lines such as "0000000000000000      DF *UND*  0000000000000000 (CXXABI_1.3.13) _ZNSt15__exception_ptr..."
    # (GNU and LLVM objdump); prints the symbols above the limits as "<version> <symbol>" and the count of
    # GLIBCXX/CXXABI references as "#count <n>"
    result=$(awk -v limits="$limits" '
        function newer(v, lim,    a, b, na, nb, i, x, y)
        {
            na = split(v, a, "."); nb = split(lim, b, ".")
            for (i = 1; i <= (na > nb ? na : nb); i++)
            {
                x = (i <= na) ? a[i] + 0 : 0; y = (i <= nb) ? b[i] + 0 : 0
                if (x != y) return x > y
            }
            return 0
        }
        BEGIN {
            n = split(limits, rows, "\n")
            for (i = 1; i <= n; i++) { split(rows[i], kv, " "); lim[kv[1]] = kv[2] }
        }
        match($0, /\((GLIBCXX|CXXABI|GCC)_[0-9]+(\.[0-9]+)*\)/) {
            tag = substr($0, RSTART + 1, RLENGTH - 2)
            p = index(tag, "_"); name = substr(tag, 1, p - 1); ver = substr(tag, p + 1)
            if (name != "GCC") count++
            if (newer(ver, lim[name])) print tag, $NF
        }
        END { print "#count", count + 0 }' <<< "$syms")

    n_versioned=$((n_versioned + $(awk '$1 == "#count" { print $2 }' <<< "$result")))
    while read -r tag sym; do
        [ -z "$tag" ] || [ "$tag" = "#count" ] && continue
        # No "--" before the name: LLVM's c++filt does not accept it, and mangled names never start with "-"
        found+="  $(basename "$f"): $tag $(c++filt "$sym" 2>/dev/null || echo "$sym")"$'\n'
    done <<< "$result"
done

[ "$n_versioned" -gt 0 ] || fail "no GLIBCXX or CXXABI symbol versions found, the output of objdump was not understood"

if [ -n "$found" ]; then
    echo "check_mex_symbol_versions: MEX files need symbol versions newer than the C++ runtime of MATLAB R2024a:"
    printf '%s' "$found"
    echo "These MEX files fail to load in MATLAB releases that bundle an older libstdc++ or libgcc_s."
    echo "Avoid the standard library feature that needs the symbol, or provide it in src/mex_compat_shim.cpp."
    exit 1
fi

echo "check_mex_symbol_versions: ${#mex_files[@]} MEX files checked, all symbol versions load in MATLAB R2024a"
exit 0
