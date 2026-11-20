#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
#
# Checks that the MATLAB / Octave MEX wrappers call the C++ library only inside CALL_QD or a try block.
#
# A C++ exception that leaves a MEX function is not turned into a MATLAB error: Octave aborts ("terminate called
# after throwing an instance of ..."), MATLAB reports an unknown exception or crashes. Every call that can throw must
# therefore run inside CALL_QD (src/qd_helpers_mex.hpp), which passes the message on as a MATLAB error, or inside a
# try block. Such a call usually throws only for unusual inputs, so the tests rarely reach it; this check finds it in
# the source.
#
# Checked are the calls of free functions quadriga_lib::<name>(...) (class names such as quadriga_lib::arrayant<double>()
# are skipped) and of the helpers defined in src/qd_helpers_wrappers.hpp plus qd_snapshot_list. Not checked are
# member functions, other helpers, std:: calls and the handlers of try blocks. A call that cannot throw may stay
# outside; mark it with the comment "// no CALL_QD: <reason>" on the line where the call starts. Raw string literals
# are not supported and make the check fail.
#
# Usage: bash tests/check_mex_call_qd.sh [<folder with the MEX sources>]   (default: api_mex)
# Fails (exit 1) when a checked call is outside CALL_QD and outside a try block. Exits with 2 when the sources cannot
# be read or were not understood. Needs awk (POSIX) and sed.

set -u

fail()
{
    echo "check_mex_call_qd: $*" >&2
    exit 2
}

usage()
{
    echo "Usage: $0 [<folder with the MEX sources>]" >&2
    exit 2
}

root="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -le 1 ] || usage
case "${1:-}" in -*) usage ;; esac
src_dir="${1:-$root/api_mex}"
[ -d "$src_dir" ] || fail "'$src_dir' is not a folder"
for tool in awk sed; do
    command -v "$tool" >/dev/null 2>&1 || fail "'$tool' not found"
done

# Class names of the public headers and the throwing wrapper helpers
include_dir="$root/include"
helpers="$root/src/qd_helpers_wrappers.hpp"
[ -d "$include_dir" ] || fail "'$include_dir' not found"
[ -f "$helpers" ] || fail "'$helpers' not found"
classes=$(sed -nE 's/^[[:space:]]*(class|struct)[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\2/p' "$include_dir"/*.hpp | sort -u | tr '\n' ' ')
# Definitions start in column 1: "static inline <type> qd_<name>(", also with a template line before
helper_names=$(sed -nE 's/^[^[:space:]#\/*(][^(]*[^A-Za-z0-9_](qd_[A-Za-z0-9_]+)[[:space:]]*\(.*/\1/p' "$helpers" | sort -u | tr '\n' ' ')
[ -n "$classes" ] || fail "no class names found in '$include_dir'"
[ -n "$helper_names" ] || fail "no helper functions found in '$helpers'"
helper_names="$helper_names qd_snapshot_list"

sources=()
for f in "$src_dir"/*.cpp; do
    [ -e "$f" ] || [ -L "$f" ] || continue # the pattern itself when nothing matched
    [ -f "$f" ] && [ -r "$f" ] || fail "'$f' is not a readable file"
    sources+=("$f")
done
[ ${#sources[@]} -gt 0 ] || fail "no .cpp files in '$src_dir'"

# Per file: blank out comments and literals (keeping line breaks), mark the ranges of CALL_QD(...) and try {...}, then
# print every checked call outside them as "<line> <name>". The last line is "#stats <n_protected> <n_calls>
# <n_marked>", or "#error <text>" when the file was not understood.
scan='
function ident_at(s, i,    j)
{
    j = i
    while (j <= n && substr(s, j, 1) ~ /[A-Za-z0-9_]/) j++
    return substr(s, i, j - i)
}
function skip_ws(s, i,    c)
{
    while (i <= n) { c = substr(s, i, 1); if (c != " " && c != "\t" && c != "\n" && c != "\r") break; i++ }
    return i
}
function match_close(s, i, op, cp,    d, c)  # i at the opening character; 0 when there is no partner
{
    d = 0
    for (; i <= n; i++)
    {
        c = substr(s, i, 1)
        if (c == op) d++
        else if (c == cp) { d--; if (d == 0) return i }
    }
    return 0
}
function skip_targs(s, j,    d, t)  # j at a possible "<"; returns the position after the template arguments, 0 if none
{
    if (substr(s, j, 1) != "<") return j
    for (d = 0; j <= n; j++)
    {
        t = substr(s, j, 1)
        if (t == "<") d++
        else if (t == ">" && substr(s, j - 1, 1) != "-") { if (--d == 0) return skip_ws(s, j + 1) }
        else if (t == ";" || t == "{") return 0
    }
    return 0
}
function protected(i,    k)
{
    for (k = 1; k <= np; k++) if (i >= ps[k] && i <= pe[k]) return 1
    return 0
}
{ raw[NR] = $0; text = text $0 "\n" }
END {
    split(classlist, cl, " "); for (k in cl) is_class[cl[k]] = 1
    split(helperlist, hl, " "); for (k in hl) is_helper[hl[k]] = 1
    dq = "\""; sq = sprintf("%c", 39); bs = "\\"

    # Blank out comments, string and character literals; positions and line breaks stay. A literal ends at the latest
    # at the end of its line, so a stray quote cannot hide the rest of the file.
    n = length(text); out = ""; state = 0; i = 1
    while (i <= n)
    {
        c = substr(text, i, 1); c2 = substr(text, i, 2)
        if (state == 0)
        {
            if (c2 == "//") { state = 1; out = out "  "; i += 2; continue }
            if (c2 == "/*") { state = 2; out = out "  "; i += 2; continue }
            if (c == "R" && substr(text, i + 1, 1) == dq && substr(text, i - 1, 1) !~ /[A-Za-z0-9_]/)
                { print "#error raw string literal in line " NR_at(i); exit }
            if (c == dq) state = 3
            else if (c == sq && substr(text, i - 1, 1) !~ /[0-9A-Fa-f]/) state = 4 # not a digit separator
            out = out c
        }
        else if (state == 1) { if (c == "\n") { state = 0; out = out c } else out = out " " }
        else if (state == 2)
        {
            if (c2 == "*/") { state = 0; out = out "  "; i += 2; continue }
            out = out (c == "\n" ? c : " ")
        }
        else
        {
            if (c == bs) { out = out " " (substr(text, i + 1, 1) == "\n" ? "\n" : " "); i += 2; continue }
            if (c == "\n") { state = 0; out = out c }
            else if (c == (state == 3 ? dq : sq)) { state = 0; out = out c }
            else out = out " "
        }
        i++
    }
    if (state == 2) { print "#error unterminated block comment"; exit }
    s = out; n = length(s)

    ln = 1
    for (i = 1; i <= n; i++) { line[i] = ln; if (substr(s, i, 1) == "\n") ln++ }

    # Pass 1: ranges of CALL_QD(...) and try {...}
    np = 0
    for (i = 1; i <= n; i++)
    {
        if (substr(s, i, 1) !~ /[A-Za-z_]/ || (i > 1 && substr(s, i - 1, 1) ~ /[A-Za-z0-9_]/)) continue
        id = ident_at(s, i); j = skip_ws(s, i + length(id))
        if ((id == "CALL_QD" && substr(s, j, 1) == "(") || (id == "try" && substr(s, j, 1) == "{"))
        {
            e = (id == "try") ? match_close(s, j, "{", "}") : match_close(s, j, "(", ")")
            if (e == 0) { print "#error unbalanced " id " in line " line[i]; exit }
            np++; ps[np] = j; pe[np] = e
        }
        i += length(id) - 1
    }

    # Pass 2: checked calls outside the ranges
    nc = 0; nm = 0
    for (i = 1; i <= n; i++)
    {
        if (substr(s, i, 1) !~ /[A-Za-z_]/ || (i > 1 && substr(s, i - 1, 1) ~ /[A-Za-z0-9_]/)) continue
        id = ident_at(s, i); start = i; i += length(id) - 1
        if (id == "quadriga_lib" && substr(s, i + 1, 2) == "::")
        {
            name = ident_at(s, i + 3)
            if (name == "") continue
            i += 2 + length(name)
            if (name in is_class) continue
        }
        else if (id in is_helper)
        {
            p = (start > 2) ? substr(s, start - 2, 2) : ""
            if (p == "::" || p == "->" || substr(p, 2, 1) == ".") continue
            name = id
        }
        else continue
        j = skip_targs(s, skip_ws(s, i + 1))
        if (j == 0 || substr(s, j, 1) != "(") continue
        nc++
        if (protected(start)) continue
        if (index(raw[line[start]], "// no CALL_QD:") > 0) { nm++; continue }
        print line[start], name
    }
    print "#stats", np, nc, nm
}
function NR_at(i,    k, c) { c = 1; for (k = 1; k < i; k++) if (substr(text, k, 1) == "\n") c++; return c }'

found=""
n_protected=0
n_calls=0
n_marked=0
for f in "${sources[@]}"; do
    result=$(awk -v classlist="$classes" -v helperlist="$helper_names" "$scan" "$f") || fail "awk cannot read '$f'"
    stats=0
    while read -r a b c d; do
        [ -z "$a" ] && continue
        case "$a" in
        "#error") fail "'$f' not understood: $b $c $d" ;;
        "#stats")
            stats=1
            n_protected=$((n_protected + b))
            n_calls=$((n_calls + c))
            n_marked=$((n_marked + d))
            ;;
        *) found+="  $(basename "$f"):$a: $b"$'\n' ;;
        esac
    done <<< "$result"
    [ "$stats" -eq 1 ] || fail "no result for '$f', the awk program was not understood"
done

if [ -n "$found" ]; then
    echo "check_mex_call_qd: C++ calls outside CALL_QD and outside a try block:"
    printf '%s' "$found"
    echo "An exception from such a call leaves the MEX function and aborts Octave or MATLAB."
    echo "Wrap the call in CALL_QD(...), or mark a call that cannot throw with \"// no CALL_QD: <reason>\"."
    exit 1
fi

# Every MEX source calls the library inside CALL_QD; finding neither means that the sources were not understood
[ "$n_calls" -gt 0 ] || fail "no library calls found in '$src_dir', the sources were not understood"
[ "$n_protected" -gt 0 ] || fail "no CALL_QD or try block found in '$src_dir', the sources were not understood"

echo "check_mex_call_qd: ${#sources[@]} MEX sources checked, $n_calls C++ calls inside CALL_QD or a try block" \
    "or marked as not throwing ($n_marked)"
exit 0
